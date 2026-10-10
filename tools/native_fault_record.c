/* Linux-only, opt-in pytest diagnostics. No production-library dependency. */
#define _GNU_SOURCE
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

#define MODULE_LIMIT 128
#define NAME_LIMIT 96
struct diagnostic_module {
  uintptr_t start, end, file_offset;
  char name[NAME_LIMIT];
};
struct module_table {
  unsigned count;
  struct diagnostic_module modules[MODULE_LIMIT];
};
#define SNAPSHOT_LIMIT 256
_Static_assert(ATOMIC_POINTER_LOCK_FREE == 2 && ATOMIC_INT_LOCK_FREE == 2,
               "signal diagnostics needs lock-free pointers and integers");
static struct module_table tables[SNAPSHOT_LIMIT];
static unsigned table_count;
static const struct module_table empty_table;
static _Atomic(const struct module_table*) active_table = &empty_table;
static atomic_int first_signal;
static int record_fd = -1;
static int installed;
static struct sigaction previous_segv, previous_abrt;

/* Serialized by the plugin RLock and PyDLL/GIL. Never called from a signal.
 * Snapshots are immutable after publication and are never reused. */
int diagnostic_set_modules(const struct diagnostic_module* modules, unsigned count) {
  if (count > MODULE_LIMIT) return -1;
  if (table_count == SNAPSHOT_LIMIT) return -2;
  unsigned next = table_count;
  tables[next].count = count;
  for (unsigned i = 0; i < count; ++i) {
    if (modules[i].start >= modules[i].end) return -1;
    tables[next].modules[i] = modules[i];
    tables[next].modules[i].name[NAME_LIMIT - 1] = '\0';
  }
  ++table_count;
  atomic_store_explicit(&active_table, &tables[next], memory_order_release);
  return 0;
}

/* Formatting below uses only bounded local arrays and integer arithmetic. */
static unsigned append_text(char* out, unsigned pos, const char* text, unsigned limit) {
  for (unsigned i = 0; i < limit && text[i] && pos < 510; ++i) out[pos++] = text[i];
  return pos;
}
static unsigned append_hex(char* out, unsigned pos, uintptr_t value) {
  const char digits[] = "0123456789abcdef";
  char reversed[2 * sizeof(uintptr_t)];
  unsigned n = 0;
  do {
    reversed[n++] = digits[value & 15U];
    value >>= 4;
  } while (value);
  out[pos++] = '0';
  out[pos++] = 'x';
  while (n) out[pos++] = reversed[--n];
  return pos;
}
static void terminate_with_signal(int sig) {
  struct sigaction action;
  action.sa_handler = SIG_DFL;
  action.sa_flags = 0;
  sigemptyset(&action.sa_mask);
  sigaction(sig, &action, NULL);
  sigset_t one;
  sigemptyset(&one);
  sigaddset(&one, sig);
  sigprocmask(SIG_UNBLOCK, &one, NULL);
  raise(sig);
  _exit(128 + sig); /* Only if the kernel could not deliver the fatal signal. */
}
static void fault_handler(int sig, siginfo_t* info, void* context) {
  int expected = 0;
  if (!atomic_compare_exchange_strong_explicit(&first_signal, &expected, sig, memory_order_relaxed,
                                               memory_order_relaxed))
    terminate_with_signal(expected);
  uintptr_t pc = 0;
#if defined(__x86_64__) && defined(REG_RIP)
  pc = (uintptr_t)((ucontext_t*)context)->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
  pc = (uintptr_t)((ucontext_t*)context)->uc_mcontext.pc;
#else
  (void)context;
#endif
  char out[512];
  unsigned pos = append_text(out, 0, "NATIVE_FAULT signal=", 32);
  pos = append_hex(out, pos, (uintptr_t)sig);
  pos = append_text(out, pos, " code=", 8);
  int code = info ? info->si_code : 0;
  if (code < 0) {
    out[pos++] = '-';
    pos = append_hex(out, pos, (uintptr_t)(-code));
  } else
    pos = append_hex(out, pos, (uintptr_t)code);
  const struct module_table* table = atomic_load_explicit(&active_table, memory_order_acquire);
  unsigned found = 0;
  for (unsigned i = 0; i < table->count; ++i) {
    const struct diagnostic_module* module = &table->modules[i];
    if (pc >= module->start && pc < module->end) {
      pos = append_text(out, pos, " module=", 16);
      pos = append_text(out, pos, module->name, NAME_LIMIT);
      pos = append_text(out, pos, " file_offset=", 16);
      pos = append_hex(out, pos, pc - module->start + module->file_offset);
      found = 1;
      break;
    }
  }
  if (!found) pos = append_text(out, pos, " module=unresolved", 32);
  out[pos++] = '\n';
  /* One small best-effort write: no malloc, stdio, loader calls or unwinder. */
  if (record_fd >= 0) {
    const ssize_t written = write(record_fd, out, pos);
    (void)written; /* Partial/error writes stay best-effort; never retry in this handler. */
  }
  const struct sigaction* previous = sig == SIGSEGV ? &previous_segv : &previous_abrt;
  sigset_t one;
  sigemptyset(&one);
  sigaddset(&one, sig);
  sigprocmask(SIG_UNBLOCK, &one, NULL);
  /* Retain our disposition while chaining, so recursive same/cross-signal
   * faults hit the guard. Python faulthandler restores its own predecessor
   * before re-raising; its original context is passed through unchanged. */
  if (previous->sa_handler != SIG_DFL && previous->sa_handler != SIG_IGN) {
    if (previous->sa_flags & SA_SIGINFO)
      previous->sa_sigaction(sig, info, context);
    else
      previous->sa_handler(sig);
  }
  /* A returning/ignored prior handler must not turn a fatal fault into a pass. */
  terminate_with_signal(sig);
}
int diagnostic_install(int fd) {
  if (fd < 0) return -1;
  if (installed) return -2;
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_sigaction = fault_handler;
  action.sa_flags = SA_SIGINFO | SA_ONSTACK;
  sigemptyset(&action.sa_mask);
  record_fd = fd;
  atomic_store_explicit(&first_signal, 0, memory_order_relaxed);
  if (sigaction(SIGSEGV, &action, &previous_segv) != 0) return -1;
  if (sigaction(SIGABRT, &action, &previous_abrt) != 0) {
    sigaction(SIGSEGV, &previous_segv, NULL);
    return -1;
  }
  installed = 1;
  return 0;
}
void diagnostic_restore(void) {
  if (!installed) return;
  struct sigaction current;
  if (sigaction(SIGSEGV, NULL, &current) == 0 && current.sa_sigaction == fault_handler)
    sigaction(SIGSEGV, &previous_segv, NULL);
  if (sigaction(SIGABRT, NULL, &current) == 0 && current.sa_sigaction == fault_handler)
    sigaction(SIGABRT, &previous_abrt, NULL);
  record_fd = -1;
  installed = 0;
}

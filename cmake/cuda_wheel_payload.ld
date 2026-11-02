/* CUDA payloads are opaque to the host; only the wrappers contain host pointers.
 * Retain every real/PTX/relocatable image, without changing compression or device
 * linking. Keeping host sections together avoids x86-64 PC32/GOTPCREL/TLSGD
 * overflows when the combined multi-architecture images exceed 2 GiB.
 */
SECTIONS
{
  .nvFatBinSegment : { KEEP(*(.nvFatBinSegment .nvFatBinSegment.*)) }
}
INSERT AFTER .data;

SECTIONS
{
  /* Start a read-only load segment instead of sharing writable host pages. */
  .nv_fatbin ALIGN(CONSTANT(MAXPAGESIZE)) : { KEEP(*(.nv_fatbin .nv_fatbin.*)) }
  __nv_relfatbin ALIGN(CONSTANT(MAXPAGESIZE)) : { KEEP(*(__nv_relfatbin __nv_relfatbin.*)) }
}
INSERT AFTER .bss;

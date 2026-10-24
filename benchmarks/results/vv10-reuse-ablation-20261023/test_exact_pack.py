"""Pure host integrity checks for full raw receipt storage; no scientific calls."""
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import exact_pack as pack


class ExactPackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, data, mode=0o644):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if os.name == 'posix':
            path.chmod(mode)
        return path

    def changed(self, raw, transform):
        value = json.loads(lzma.decompress(raw))
        transform(value)
        return lzma.compress(json.dumps(value).encode())

    def test_legacy_text_exact_lf_crlf_unicode_empty_and_mode(self):
        values = {'lf.txt': b'line\n', 'crlf.txt': b'line\r\n',
                  'unicode.txt': '\u5316\u5b66\n'.encode(), 'empty.txt': b''}
        for name, data in values.items():
            self.write(name, data, 0o755)
        raw = pack.encode(self.source, values)
        self.assertEqual(json.loads(lzma.decompress(raw))['schema'], pack.SCHEMA)
        out = self.root / 'restored'
        pack.restore(raw, out)
        for name, data in values.items():
            self.assertEqual((out / name).read_bytes(), data)
            if os.name == 'posix':
                self.assertEqual((out / name).stat().st_mode & 0o777, 0o755)
        with self.assertRaises((ValueError, FileExistsError)):
            pack.restore(raw, out)

    def test_original_npz_container_bytes_preserved(self):
        # A valid NumPy .npy member with -0.0 and a quiet NaN bit pattern.
        header = repr({'descr': '<f8', 'fortran_order': False, 'shape': (2,)})
        padding = ' ' * ((64 - (10 + len(header) + 1) % 64) % 64)
        header = (header + padding + '\n').encode()
        npy = b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header
        npy += bytes.fromhex('0000000000000080010000000000f87f')
        container = io.BytesIO()
        with zipfile.ZipFile(container, 'w') as archive:
            archive.writestr('energy.npy', npy)
        data = container.getvalue()
        self.write('sample-0/returned-arrays.npz', data)
        self.write('sample-0/returned.json', b'{"status":"RETURNED_UNVALIDATED"}\n')
        raw = pack.encode(self.source, ['sample-0/returned-arrays.npz', 'sample-0/returned.json'])
        self.assertEqual(json.loads(lzma.decompress(raw))['schema'], pack.BINARY_SCHEMA)
        pack.restore(raw, self.root / 'out')
        restored = (self.root / 'out/sample-0/returned-arrays.npz').read_bytes()
        self.assertEqual(restored, data)
        self.assertEqual(hashlib.sha256(restored).digest(), hashlib.sha256(data).digest())
        with zipfile.ZipFile(io.BytesIO(restored)) as archive:
            self.assertEqual(archive.read('energy.npy'), npy)

    def test_tampering_and_ambiguous_encodings_rejected(self):
        self.write('receipt', b'\xff\x00\n')
        raw = pack.encode(self.source, ['receipt'])
        mutations = [
            lambda v: v['members'][0].update(sha256='0' * 64),
            lambda v: v['members'][0].update(binary_base64='!!!!'),
            lambda v: v['members'][0].update(utf8='also present'),
            lambda v: v['members'][0].update(mode=True),
            lambda v: v.update(schema=pack.SCHEMA),
            lambda v: v['members'].append(v['members'][0]),
            lambda v: v['members'][0].update(path='../escaped'),
            lambda v: v['members'][0].update(path='a//b'),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises((ValueError, TypeError)):
                    pack.decode(self.changed(raw, mutation))

    def test_no_clobber_preflights_entire_destination(self):
        self.write('first', b'first')
        self.write('second', b'second')
        raw = pack.encode(self.source, ['first', 'second'])
        target = self.root / 'out'
        target.mkdir()
        (target / 'second').write_bytes(b'original')
        with self.assertRaises((ValueError, FileExistsError)):
            pack.restore(raw, target)
        self.assertFalse((target / 'first').exists())
        self.assertEqual((target / 'second').read_bytes(), b'original')

    @unittest.skipUnless(os.name == 'posix', 'POSIX symlink contract')
    def test_symlink_ancestors_refused(self):
        self.write('inside/receipt', b'value')
        (self.source / 'alias').symlink_to(self.source / 'inside', target_is_directory=True)
        with self.assertRaises(ValueError):
            pack.encode(self.source, ['alias/receipt'])
        raw = pack.encode(self.source, ['inside/receipt'])
        target = self.root / 'out'
        target.mkdir()
        actual = target / 'actual'
        actual.mkdir()
        (target / 'inside').symlink_to(actual, target_is_directory=True)
        with self.assertRaises(ValueError):
            pack.restore(raw, target)
        self.assertFalse((actual / 'receipt').exists())

    def test_optimized_python_refuses_before_reading_evidence(self):
        marker = self.root / 'unexpected-evidence-io'
        code = 'import exact_pack\nfrom pathlib import Path\nPath(' + repr(str(marker)) + ').touch()'
        env = dict(os.environ)
        env.pop('PYTHONOPTIMIZE', None)
        result = subprocess.run([sys.executable, '-O', '-c', code],
                                cwd=Path(pack.__file__).parent, env=env,
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('optimized Python is unsupported', result.stderr)
        self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()

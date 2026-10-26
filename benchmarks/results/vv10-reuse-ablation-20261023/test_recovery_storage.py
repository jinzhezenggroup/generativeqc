"""Host-only conservation checks for whole-file receipt aliases and inventory."""
import copy
import json
import lzma
import os
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

# The published source capsule contains complete, checksum-bound helpers.
# Local development can import the same ordinary files directly.
_capsule = Path(__file__).with_name('offline-source.json.xz')
if _capsule.is_file():
    import exact_pack
    _source_directory = tempfile.TemporaryDirectory(prefix='vv10_storage_test_sources_')
    exact_pack.restore(_capsule.read_bytes(), _source_directory.name)
    sys.path.insert(0, _source_directory.name)

from collect_recovery import collect, deduplicate, inventory
from exact_pack import decode, encode
from recovery_contract import sha
from reproduce_recovery import restore_collection, restore_tools, verify_source_identity
from repack_collection import repack


class RecoveryStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self):
        collection = self.root / 'collection'
        collection.mkdir()
        inventories, packs = {}, []
        original = b'{"role":"HOST_TEST_NO_SCI"}\r\n'
        for mount, name, raw in [('control', 'output/reference.json', original),
                                 ('recovery', 'spec.json', b'{"role":"HOST_TEST_NO_SCI"}\n')]:
            source = self.raw / mount
            path = source / name
            path.parent.mkdir(parents=True)
            path.write_bytes(raw)
            packed = encode(source, [name])
            members = decode(packed)
            filename = mount + '.json.xz'
            (collection / filename).write_bytes(packed)
            packs.append({'file': filename, 'mount': mount, 'packed_bytes': len(packed),
                          'sha256': sha(collection / filename)})
            inventories[mount] = {'included': [{k: v for k, v in row.items() if k != 'utf8'} for row in members], 'excluded': []}
        row = copy.deepcopy(inventories['control']['included'][0])
        row['path'] = 'legacy-output/reference.json'
        inventories['recovery']['included'].append(row)
        (collection / 'raw-inventory.json').write_text(json.dumps(inventories))
        (collection / 'numerical-replay.json').write_text('{"status":"HOST_ONLY_NO_SCI"}')
        metadata = {'schema': 'generativeqc.vv10-mixed-protocol-collection.v1', 'packs': packs,
                    'aliases': [{'mount': 'recovery', **row, 'source_mount': 'control', 'source_path': 'output/reference.json'}],
                    'inventory_sha256': sha(collection / 'raw-inventory.json'),
                    'numerical_replay_sha256': sha(collection / 'numerical-replay.json')}
        (collection / 'collection.json').write_text(json.dumps(metadata))
        return collection, metadata, original

    def test_compact_source_binds_revision_and_original_sdk_manifest(self):
        publication = Path(__file__).resolve().parent
        source = json.loads((publication / 'source-identity.json').read_bytes())
        raw = decode((publication / 'recovery-execution-and-failure-history.json.xz').read_bytes())
        spec = json.loads(next(row['utf8'] for row in raw if row['path'] == 'spec.json'))
        expected = spec['expected_sdk']
        self.assertEqual(verify_source_identity(publication, expected), source)
        trial = self.root / 'source-identity.json'
        changes = [
            ('schema', 'wrong'), ('revision', '0' * 40), ('tree', 'invalid-tree'),
            ('source_entries', source['source_entries'] - 1),
            ('source_patch', 'unexpected.patch'),
        ]
        for key, value in changes:
            altered = copy.deepcopy(source)
            altered[key] = value
            trial.write_text(json.dumps(altered))
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_source_identity(self.root, expected)
        for key, value in [('sha256', '0' * 64), ('bytes', 1), ('path', 'different.json')]:
            altered = copy.deepcopy(source)
            altered['original_manifest'][key] = value
            trial.write_text(json.dumps(altered))
            with self.subTest(manifest=key), self.assertRaises(ValueError):
                verify_source_identity(self.root, expected)
        altered = copy.deepcopy(source)
        altered['solver']['path'] = 'different.py'
        trial.write_text(json.dumps(altered))
        with self.assertRaises(ValueError):
            verify_source_identity(self.root, expected)

    def test_changed_source_identity_is_rejected_before_restoration(self):
        collection, metadata, _ = self.fixture()
        path = collection / 'source-identity.json'
        path.write_text('{"role":"HOST_ONLY_SOURCE_IDENTITY"}')
        metadata['source_identity'] = {'file': path.name, 'sha256': sha(path)}
        (collection / 'collection.json').write_text(json.dumps(metadata))
        path.write_text('{"role":"CHANGED_SOURCE_IDENTITY"}')
        destination = self.root / 'must-not-exist'
        with self.assertRaisesRegex(ValueError, 'source identity changed'):
            restore_collection(collection, destination)
        self.assertFalse(destination.exists())

    def test_exact_whole_file_alias_restores_original_paths_bytes_and_modes(self):
        collection, _, original = self.fixture()
        target = self.root / 'restored'
        restore_collection(collection, target)
        self.assertEqual((target / 'control/output/reference.json').read_bytes(), original)
        self.assertEqual((target / 'recovery/legacy-output/reference.json').read_bytes(), original)
        with self.assertRaises(ValueError):
            restore_collection(collection, target)

    def test_alias_corruption_and_escape_fail_closed(self):
        collection, metadata, _ = self.fixture()
        for change in ({'sha256': '0' * 64}, {'path': '../escape'}):
            altered = copy.deepcopy(metadata)
            altered['aliases'][0].update(change)
            (collection / 'collection.json').write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                restore_collection(collection, self.root / ('attempt-' + str(len(change['sha256']) if 'sha256' in change else 'escape')))

    def test_metadata_compression_preserves_original_byte_hashes(self):
        collection, _, original = self.fixture()
        for name in ('collection.json', 'raw-inventory.json', 'numerical-replay.json'):
            source = collection / name
            (collection / (name + '.xz')).write_bytes(lzma.compress(source.read_bytes()))
            source.unlink()  # Only this test's newly generated temporary fixture.
        target = self.root / 'restored-compressed'
        restore_collection(collection, target)
        self.assertEqual((target / 'recovery/legacy-output/reference.json').read_bytes(), original)

    def test_recompression_keeps_expanded_bytes_and_original_collection(self):
        collection, metadata, original = self.fixture()
        metadata['total_packed_bytes'] = sum(row['packed_bytes'] for row in metadata['packs'])
        (collection / 'collection.json').write_text(json.dumps(metadata))
        before = {path.name: path.read_bytes() for path in collection.iterdir()}
        compact = self.root / 'repacked'
        repack(collection, compact)
        after = {path.name: path.read_bytes() for path in collection.iterdir()}
        self.assertEqual(before, after)
        for row in metadata['packs']:
            self.assertEqual(lzma.decompress((compact / row['file']).read_bytes()),
                             lzma.decompress(before[row['file']]))
        target = self.root / 'restored-repacked'
        restore_collection(compact, target)
        self.assertEqual((target / 'recovery/legacy-output/reference.json').read_bytes(), original)

    def test_offline_sources_are_complete_and_hash_bound(self):
        source = self.raw / 'sources'
        source.mkdir()
        (source / 'helper.py').write_bytes(b'VALUE = "HOST_TEST_NO_SCI"\n')
        raw = encode(source, ['helper.py'])
        publication = self.root / 'publication'
        publication.mkdir()
        (publication / 'offline-source.json.xz').write_bytes(raw)
        manifest = {'files': [{'path': 'offline-source.json.xz', 'role': 'reproduction'}],
                    'offline_source_pack': {'file': 'offline-source.json.xz', 'packed_bytes': len(raw),
                                             'sha256': sha(publication / 'offline-source.json.xz')}}
        target = restore_tools(publication, self.root / 'offline-output', manifest)
        self.assertEqual((target / 'helper.py').read_bytes(), (source / 'helper.py').read_bytes())
        (publication / 'offline-source.json.xz').write_bytes(raw + b'HOST_CORRUPTION')
        with self.assertRaises(ValueError):
            restore_tools(publication, self.root / 'never-written', manifest)

    def test_inventory_retains_real_npz_and_indexes_compiled_objects(self):
        with (self.raw / 'returned-arrays.npz').open('xb') as stream:
            np.savez(stream, energy=np.asarray(-0.0), forces=np.zeros((3, 3)), endpoint_seconds=np.asarray(1.0))
        (self.raw / 'runtime.so').write_bytes(b'\x7fELF' + b'HOST_COMPILED_FIXTURE')
        included, excluded = inventory(self.raw)
        self.assertEqual([row['path'] for row in included], ['returned-arrays.npz'])
        self.assertEqual([row['path'] for row in excluded], ['runtime.so'])
        (self.raw / 'disguised.json').write_bytes(b'\x7fELF' + b'HOST_COMPILED_FIXTURE')
        with self.assertRaisesRegex(ValueError, 'ELF disguised'):
            inventory(self.raw)

    def test_unknown_file_is_not_silently_dropped(self):
        (self.raw / 'unknown.payload').write_bytes(b'possibly important')
        with self.assertRaisesRegex(ValueError, 'unclassified'):
            inventory(self.raw)

    def test_empty_preparation_markers_are_preserved_without_scientific_meaning(self):
        receipts = self.raw / 'receipts'
        receipts.mkdir()
        for name in ('build.complete', 'runtime.complete'):
            (receipts / name).touch()
        included, _ = inventory(self.raw, sdk=True)
        self.assertEqual({row['path'] for row in included}, {'receipts/build.complete', 'receipts/runtime.complete'})
        self.assertTrue(all(row['bytes'] == 0 for row in included))
        (receipts / 'build.complete').write_bytes(b'unexpected')
        with self.assertRaisesRegex(ValueError, 'unexpectedly contains payload'):
            inventory(self.raw, sdk=True)

    def test_whole_file_dedup_conserves_paths_and_different_modes(self):
        canonical = {}
        first = {'path': 'a', 'bytes': 2, 'sha256': 'a' * 64, 'mode': 0o644}
        stored, aliases = deduplicate('control', [first], canonical)
        self.assertEqual(stored, [first])
        self.assertEqual(aliases, [])
        duplicate = {**first, 'path': 'b', 'mode': 0o755}
        distinct = {**first, 'path': 'c', 'sha256': 'b' * 64}
        stored, aliases = deduplicate('recovery', [duplicate, distinct], canonical)
        self.assertEqual(stored, [distinct])
        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0]['source_mount'], 'control')
        self.assertEqual(aliases[0]['source_path'], 'a')
        self.assertEqual(aliases[0]['path'], 'b')
        self.assertEqual(aliases[0]['mode'], 0o755)

    def test_object_npz_rejected(self):
        with (self.raw / 'bad.npz').open('xb') as stream:
            np.savez(stream, energy=np.asarray(object()), forces=np.zeros((1, 3)), endpoint_seconds=np.asarray(1.0))
        with self.assertRaises(ValueError):
            inventory(self.raw)

    def test_nonterminal_collection_never_creates_destination(self):
        roots = []
        for name in ('sdk', 'control', 'recovery', 'diagnostic'):
            root = self.raw / name
            root.mkdir()
            roots.append(root)
        destination = self.root / 'never-created'
        with self.assertRaises(FileNotFoundError):
            collect(*roots, destination)
        self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()

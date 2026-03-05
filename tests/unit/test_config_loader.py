"""
Unit tests for config.config_loader — env-var interpolation, overlay, deep merge.
"""
import os
import tempfile
import unittest

import yaml

# Ensure monai_aegis is on the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'monai_aegis'))

from config.config_loader import resolve_env_vars, deep_merge, load_config


class TestResolveEnvVars(unittest.TestCase):
    """Test ${VAR:default} resolution."""

    def test_resolves_set_env_var(self):
        os.environ['AEGIS_TEST_VAR'] = '/mnt/s3/bucket'
        try:
            result = resolve_env_vars('${AEGIS_TEST_VAR}')
            self.assertEqual(result, '/mnt/s3/bucket')
        finally:
            del os.environ['AEGIS_TEST_VAR']

    def test_uses_default_when_unset(self):
        os.environ.pop('AEGIS_UNSET_VAR', None)
        result = resolve_env_vars('${AEGIS_UNSET_VAR:staging_input}')
        self.assertEqual(result, 'staging_input')

    def test_raises_when_no_default_and_unset(self):
        os.environ.pop('AEGIS_MISSING_VAR', None)
        with self.assertRaises(KeyError) as ctx:
            resolve_env_vars('${AEGIS_MISSING_VAR}')
        self.assertIn('AEGIS_MISSING_VAR', str(ctx.exception))

    def test_resolves_in_nested_dict(self):
        os.environ['AEGIS_TEST_NESTED'] = 'prod_value'
        try:
            data = {'level1': {'level2': '${AEGIS_TEST_NESTED}'}}
            result = resolve_env_vars(data)
            self.assertEqual(result['level1']['level2'], 'prod_value')
        finally:
            del os.environ['AEGIS_TEST_NESTED']

    def test_resolves_in_list(self):
        os.environ['AEGIS_TEST_LIST'] = 'item_resolved'
        try:
            data = ['${AEGIS_TEST_LIST}', 'plain']
            result = resolve_env_vars(data)
            self.assertEqual(result, ['item_resolved', 'plain'])
        finally:
            del os.environ['AEGIS_TEST_LIST']

    def test_non_string_values_unchanged(self):
        data = {'count': 42, 'enabled': True, 'rate': 0.5, 'empty': None}
        result = resolve_env_vars(data)
        self.assertEqual(result, data)

    def test_mixed_text_and_env_var(self):
        os.environ['AEGIS_TEST_MIX'] = 'bucket-123'
        try:
            result = resolve_env_vars('s3://${AEGIS_TEST_MIX}/data')
            self.assertEqual(result, 's3://bucket-123/data')
        finally:
            del os.environ['AEGIS_TEST_MIX']

    def test_multiple_vars_in_one_string(self):
        os.environ['AEGIS_A'] = 'alpha'
        os.environ['AEGIS_B'] = 'beta'
        try:
            result = resolve_env_vars('${AEGIS_A}-${AEGIS_B}')
            self.assertEqual(result, 'alpha-beta')
        finally:
            del os.environ['AEGIS_A']
            del os.environ['AEGIS_B']

    def test_env_var_overrides_default(self):
        os.environ['AEGIS_OVERRIDE'] = 'env_wins'
        try:
            result = resolve_env_vars('${AEGIS_OVERRIDE:default_value}')
            self.assertEqual(result, 'env_wins')
        finally:
            del os.environ['AEGIS_OVERRIDE']

    def test_empty_default(self):
        os.environ.pop('AEGIS_EMPTY_DEFAULT', None)
        result = resolve_env_vars('${AEGIS_EMPTY_DEFAULT:}')
        self.assertEqual(result, '')


class TestDeepMerge(unittest.TestCase):
    """Test deep-merge overlay logic."""

    def test_simple_override(self):
        base = {'a': 1, 'b': 2}
        overlay = {'b': 3}
        result = deep_merge(base, overlay)
        self.assertEqual(result, {'a': 1, 'b': 3})

    def test_nested_merge(self):
        base = {'paths': {'input': 'a', 'output': 'b'}, 'ocr': {'gpu': False}}
        overlay = {'paths': {'input': 'new_a'}}
        result = deep_merge(base, overlay)
        self.assertEqual(result['paths']['input'], 'new_a')
        self.assertEqual(result['paths']['output'], 'b')  # preserved
        self.assertEqual(result['ocr']['gpu'], False)       # preserved

    def test_new_keys_added(self):
        base = {'a': 1}
        overlay = {'b': 2}
        result = deep_merge(base, overlay)
        self.assertEqual(result, {'a': 1, 'b': 2})

    def test_inputs_not_mutated(self):
        base = {'a': {'x': 1}}
        overlay = {'a': {'x': 99}}
        result = deep_merge(base, overlay)
        self.assertEqual(base['a']['x'], 1)  # original unchanged
        self.assertEqual(result['a']['x'], 99)


class TestLoadConfig(unittest.TestCase):
    """Test end-to-end config loading."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_yaml(self, data, filename='config.yaml'):
        path = os.path.join(self.tmpdir, filename)
        with open(path, 'w') as f:
            yaml.dump(data, f)
        return path

    def test_loads_base_config(self):
        path = self._write_yaml({
            'paths': {
                'input': '${AEGIS_TEST_LOAD:default_in}',
                'output': 'fixed_out',
            }
        })
        os.environ.pop('AEGIS_TEST_LOAD', None)
        config = load_config(path)
        self.assertEqual(config['paths']['input'], 'default_in')
        self.assertEqual(config['paths']['output'], 'fixed_out')

    def test_overlay_merges(self):
        base_path = self._write_yaml({
            'paths': {'input': 'base_in', 'output': 'base_out'},
            'ocr': {'gpu': False},
        })
        overlay_path = self._write_yaml(
            {'paths': {'input': 'overlay_in'}},
            filename='overlay.yaml',
        )
        config = load_config(base_path, overlay_path=overlay_path)
        self.assertEqual(config['paths']['input'], 'overlay_in')
        self.assertEqual(config['paths']['output'], 'base_out')
        self.assertFalse(config['ocr']['gpu'])

    def test_aegis_config_override_env(self):
        base_path = self._write_yaml({'mode': 'base'})
        overlay_path = self._write_yaml({'mode': 'prod'}, filename='prod.yaml')

        os.environ['AEGIS_CONFIG_OVERRIDE'] = overlay_path
        try:
            config = load_config(base_path)
            self.assertEqual(config['mode'], 'prod')
        finally:
            del os.environ['AEGIS_CONFIG_OVERRIDE']

    def test_env_vars_resolved_after_overlay(self):
        base_path = self._write_yaml({
            'paths': {'input': '${AEGIS_LATE_RESOLVE:early_default}'}
        })
        overlay_path = self._write_yaml(
            {'paths': {'input': '${AEGIS_LATE_RESOLVE:late_default}'}},
            filename='overlay.yaml',
        )
        os.environ.pop('AEGIS_LATE_RESOLVE', None)
        config = load_config(base_path, overlay_path=overlay_path)
        # Overlay value wins, then resolved with default
        self.assertEqual(config['paths']['input'], 'late_default')

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config('/nonexistent/config.yaml')


if __name__ == '__main__':
    unittest.main()

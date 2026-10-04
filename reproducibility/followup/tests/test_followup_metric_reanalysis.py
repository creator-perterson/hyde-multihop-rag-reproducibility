"""Discriminating fail-closed public-vector contract tests."""
import json
import hashlib
import tempfile
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.reanalyze_jiis_followup_metrics import cells, compare, metric_rows, verify_manifest

class PublicVectorTests(unittest.TestCase):
    def test_reordered_pair_is_rejected(self):
        r=[{'id':'a','method':'base'},{'id':'b','method':'base'},
           {'id':'b','method':'target'},{'id':'a','method':'target'}]
        with self.assertRaisesRegex(ValueError,'ordering'):
            cells(r,('method',),n=2,id_key='id')

    def test_duplicate_id_is_rejected(self):
        r=[{'id':'a','method':'base'},{'id':'a','method':'base'}]
        with self.assertRaisesRegex(ValueError,'duplicate'):
            cells(r,('method',),n=2,id_key='id')

    def test_missing_cell_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            cells([{'id':'a','method':'base'}],('method',),n=2,id_key='id')

    def test_changed_frozen_statistic_is_rejected(self):
        a=[{'family':'em','p_holm':.02}];e=[{'family':'em','p_holm':'.03'}]
        with self.assertRaisesRegex(ValueError,'numeric mismatch'):
            compare(a,e,('family',))

    def test_summary_order_is_rejected(self):
        a=[{'family':'a','delta':0},{'family':'b','delta':0}]
        e=[{'family':'b','delta':'0'},{'family':'a','delta':'0'}]
        with self.assertRaisesRegex(ValueError,'order'):
            compare(a,e,('family',))

    def test_input_tamper_against_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);vector=root/'metrics.csv';vector.write_text('id,em\na,1\n')
            digest=hashlib.sha256(vector.read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps({'files':[{'path':'metrics.csv','sha256':digest}]}))
            self.assertTrue(verify_manifest(root))
            vector.write_text('id,em\na,0\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_manifest(root)

    def test_private_and_nonfinite_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'metrics.csv'
            p.write_text('id,em,prediction\na,1,private text\n')
            with self.assertRaisesRegex(ValueError,'nonpublic'):
                metric_rows(p)
            p.write_text('id,em\na,nan\n')
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                metric_rows(p)

if __name__=='__main__':unittest.main()

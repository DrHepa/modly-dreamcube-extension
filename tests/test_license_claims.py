from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


EXT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = EXT_DIR / "manifest.json"
VALIDATOR_PATH = EXT_DIR / "validate_extension.py"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("dreamcube_validate_extension", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LicenseClaimsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.readme = (EXT_DIR / "README.md").read_text(encoding="utf-8")
        cls.notices = (EXT_DIR / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        cls.validator_module = load_validator_module()

    def test_manifest_distinguishes_wrapper_source_and_both_weight_licenses(self):
        self.assertEqual(
            self.manifest["license"],
            {
                "wrapper": "MIT",
                "upstream": "NOASSERTION",
                "weights": "Apache-2.0",
                "auto_depth_weights": "Apache-2.0",
            },
        )

    def test_validator_rejects_apache_claim_for_upstream_source(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["license"]["upstream"] = "Apache-2.0"
        validator = self.validator_module.Validator()

        self.validator_module.validate_manifest(validator, manifest)

        self.assertIn(
            "license.upstream: expected 'NOASSERTION', got 'Apache-2.0'",
            validator.errors,
        )

    def test_docs_state_source_license_is_not_declared(self):
        for document in (self.readme, self.notices):
            self.assertIn("NOASSERTION", document)
            self.assertIn("source-code license", document)
            self.assertIn("not declared", document)

        self.assertIsNone(
            re.search(
                r"DreamCube source code.*licensed under.*Apache",
                self.notices,
                flags=re.IGNORECASE,
            )
        )
        self.assertIsNone(
            re.search(
                r"DreamCube upstream source.*weights.*declared under Apache",
                self.readme,
                flags=re.IGNORECASE,
            )
        )

    def test_docs_scope_mit_to_wrapper_and_apache_to_each_weight_repository(self):
        self.assertIn("MIT license applies only to the wrapper", self.readme)
        self.assertIn("does not relicense DreamCube source code", self.readme)
        self.assertIn("KevinHuang/DreamCube", self.readme)
        self.assertIn("depth-anything/Depth-Anything-V2-Small-hf", self.readme)
        self.assertGreaterEqual(self.readme.count("Apache-2.0"), 2)

        self.assertIn("MIT License in [LICENSE](LICENSE) applies only to the wrapper code", self.notices)
        self.assertIn("does not supply a license for the separate DreamCube source checkout", self.notices)
        self.assertIn("License declared by the model repository for its weights: Apache-2.0", self.notices)
        self.assertEqual(
            self.notices.count("License declared by the model repository for its weights: Apache-2.0"),
            2,
        )


if __name__ == "__main__":
    unittest.main()

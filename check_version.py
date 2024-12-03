"""script used in github actions to bump the patch number of the python package if it already exists"""
import fileinput
import re
import requests
import pathlib

import prescyent


REGEX = r"(__version__\s?=\s?[\"']\d*\.\d*\.)(\d*)([\"'])"

module_version = prescyent.__version__
response = requests.get(f"http://pypi.python.org/pypi/prescyent/{module_version}", timeout=5)
version_updated = False
if response.status_code == 200:
    curr_dir = pathlib.Path(__file__).parent.resolve()
    with fileinput.input(str(curr_dir / "prescyent" / "__init__.py"), inplace=True) as f:
        for line in f:
            # modify the line as needed
            version_search = re.search(REGEX, line, re.IGNORECASE)
            if version_search:
                patch_number = int(version_search.group(2)) + 1
                version_substitute = f"\\g<1>{patch_number}\\g<3>"
                print(re.sub(REGEX, version_substitute, line, 0), end='')
                version_updated = True
            else:
                print(line, end='')
if version_updated:
    print(f"library's version got updated to x.x.{patch_number}")
    exit(1)

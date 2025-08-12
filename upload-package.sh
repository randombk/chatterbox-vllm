#!/bin/bash
python3 -m build --sdist .
twine upload dist/chatterbox_vllm-$(cat .latest-version.generated.txt).tar.gz

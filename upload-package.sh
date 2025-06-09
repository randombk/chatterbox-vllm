#!/bin/bash
python3 -m build --sdist .
twine upload dist/chatterbox-vllm-$(cat .latest-version.generated.txt).tar.gz

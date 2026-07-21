#!/bin/bash

python rww_with_IM.py --measure="FC"
python rww_with_IM.py --measure="TE"
python rww_with_IM.py --measure="CD"
python rww_with_IM.py --measure="S"
python rww_with_IM.py --measure="R"
python rww_with_IM.py --measure="WMS"
python rww_with_IM.py --measure="PhiG"
python rww_with_IM.py --measure="PhiR"
python rww_with_IM.py --measure="Oinfo" --order=3
python rww_with_IM.py --measure="Oinfo" --order=4
python rww_with_IM.py --measure="Oinfo" --order=5
python rww_with_IM.py --measure="Oinfo" --order=6
python rww_with_IM.py --measure="Oinfo" --order=7
python rww_with_IM.py --measure="Oinfo" --order=8
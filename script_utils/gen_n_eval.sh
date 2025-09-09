CONFIG_NAME="inference_ucond_notri"

rm -r inference  # This removes all your inference runs so far in that directory
python proteinfoundation/generate.py --config_name $CONFIG_NAME
python proteinfoundation/evaluate.py --config_name $CONFIG_NAME

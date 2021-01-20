python utils/make_csv_file.py
python utils/make_configs.py --model SSD

files="./result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        python train.py "${filepath}/config.yaml"
        # python evaluate.py "${filepath}/config.yaml" validation
        # python evaluate.py "${filepath}/config.yaml" test
    fi
done

# python src/utils/make_final_result.py result/

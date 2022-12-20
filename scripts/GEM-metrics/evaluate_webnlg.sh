for lang in en ru ;
do
    python wrap-outputs.py $lang /tmp/webnlg.en-$lang
    python run_metrics.py -r test_data/webnlg_${lang}_test.json test_data/outs.json --metric-list rouge
done;

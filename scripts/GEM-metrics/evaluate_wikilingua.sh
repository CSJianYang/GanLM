for lang in es ru vi tr ;
do
    python wrap-outputs.py $lang /tmp/wikilingua.$lang-en
    python run_metrics.py -r test_data/wiki_lingua_${lang}_validation.json test_data/outs.json --metric-list rouge
done;

var=`sh ./find_ip.sh`
echo "remote ip:  $var"
python eval_mini_srcgame.py -t $var

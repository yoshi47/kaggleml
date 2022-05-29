# 訓練データとテストデータを1つのデータに統合するコードです。
def clean_data(fi, fo, header, suffix):
    
    # fi : 訓練／テストデータを読み込む file iterator
    # fo : 統合されるデータが writeされるパス
    # header : データに header 行を追加するかどうかを決定する boolean
    # suffix : 訓練データには 48個の変数があり、テストデータには24個の変数があります。
    # suffixで不足するテストデータ 24個分を空白で埋めます。
    # csvの最初の行、つまり headerを読み込みます。
    head = fi.readline().strip("\n").split(",")
    head = [h.strip('"') for h in head]

    # ‘‘nomprov’ 変数の位置を ipに保存します。
    for i, h in enumerate(head):
        if h == "nomprov":
            ip = i

    # headerが True である場合は、保存ファイルの headerを writeします。
    if header:
        fo.write("%s\n" % ",".join(head))

    # nは読み込んだ変数の個数を意味します。(訓練データ：48、テストデータ：24)
    n = len(head)
    for line in fi:
        # ファイルの内容を1行ずつ読み出し、改行(\n)と ‘,’で分離します。
        fields = line.strip("\n").split(",")

        # ‘‘nomprov’変数に ‘,’を含むデータが存在します。
        # ‘,’で分離されたデータを再び組み合わせます。
        if len(fields) > n:
            prov = fields[ip] + fields[ip+1]
            del fields[ip]
            fields[ip] = prov

        # データの個数が nと同じかどうかを確認し、ファイルに writeします。
        # テストデータの場合、suffixは 24個の空白です。
        assert len(fields) == n
        fields = [field.strip() for field in fields]
        fo.write("%s%s\n" % (",".join(fields), suffix))

# 1つのデータとして統合するコードを実行します。まず訓練データを writeし、その次にテストデータを writeします。これ以後1つの dataframeだけを取り扱い、前処理を進めます。
with open("../input/8th.clean.all.csv", "w") as f:
    clean_data(open("../input/train_ver2.csv"), f, True, "")
    comma24 = "".join(["," for i in range(24)])
    clean_data(open("../input/test_ver2.csv"), f, False, comma24)

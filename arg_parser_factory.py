from argparse import ArgumentParser, RawTextHelpFormatter


def build():
    parser = ArgumentParser(
        description='訓練 Work2Vec，流程為：cwiki -> segment -> train'
        , formatter_class=RawTextHelpFormatter)

    subcmd = parser.add_subparsers(
        dest='subcmd', help='subcommands', metavar='SUBCOMMAND')
    subcmd.required = True

    # 進行wiki to txt解析
    kcm_parser = subcmd.add_parser('cwiki',
                                   help='進行wiki to txt解析')
    kcm_parser.add_argument('-i',
                            dest='input',
                            help='解析的檔案路徑')
    kcm_parser.add_argument('-o',
                            dest='output',
                            help='輸出檔案路徑')

    # 進行 segment
    segment_parser = subcmd.add_parser('segment',
                                       help='進行分詞')
    segment_parser.add_argument('-i',
                                dest='input',
                                help='解析的檔案路徑')
    segment_parser.add_argument('-o',
                                dest='output',
                                help='輸出檔案路徑')

    # 進行Word2Vec
    train_parser = subcmd.add_parser('train',
                                     help='進行詞向量解析')
    train_parser.add_argument('-i',
                              dest='input',
                              help='解析的檔案路徑')
    train_parser.add_argument('-o',
                              dest='output',
                              help='輸出檔案路徑')

    # 進行查詢
    query_parser = subcmd.add_parser('query',
                                     help='進行關鍵字查詢')
    query_parser.add_argument('-i',
                              dest='input',
                              help='進行【查詢】的檔案路徑')
    query_parser.add_argument("-k", '--keyword',
                              dest='keyword',
                              help='關鍵字',
                              default="")
    query_parser.add_argument("-l", '--limit',
                              dest='limit',
                              help='篩選筆數(default:100)',
                              default=100)

    # 進行正反向查詢
    querynp_parser = subcmd.add_parser('querynp',
                                       help='進行正反向查詢')
    querynp_parser.add_argument('-ip',
                                dest='inputp',
                                help='正向 詞向量模型路徑')
    querynp_parser.add_argument('-in',
                                dest='inputn',
                                help='反向 詞向量模型路徑')
    querynp_parser.add_argument("-k", '--keyword',
                                dest='keyword',
                                help='關鍵字',
                                default="")

    return parser.parse_args()

import arg_parser_factory
from train_service import query, wiki_to_txt, segment, train, query2
import logging_utils


def main():
    args = arg_parser_factory.build()

    if args.subcmd == 'cwiki':
        wiki_to_txt(args.input, args.output)

    if args.subcmd == 'segment':
        segment(args.input, args.output)

    if args.subcmd == 'train':
        train(args.input, args.output)

    if args.subcmd == 'query':
        query(args.input, args.keyword, args.limit)

    if args.subcmd == 'querynp':
        query2(args.inputp,args.inputn, args.keyword)


if __name__ == '__main__':
    logging_utils.Init_logging()

    main()

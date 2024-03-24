import argparse
import Train
import Test


def main():
    argparser = argparse.ArgumentParser(
        description="Network for predicting topology of membrane proteins.")

    #argparser.add_argument('-p','--param', default='/zhome/be/1/138857/special_project/data/parameters.pkl',
    #                       type=str, metavar='PARAMETERS', help='path to pickle file containing parameters for current run')

    argparser.add_argument('-r','--run', default='/zhome/be/1/138857/special_project/data/run_unspecified/',
                           type=str, metavar='RUN', help='path for directory for current run')

    argparser.add_argument('--mode', default='train', type=str,
                           choices=['train', 'test'])

    #argparser.add_argument('--model_num', default='', type=str, help = 'number to indicate which training procedure is taking place')

    args = argparser.parse_args()
    if args.mode == 'train':
        Train.train(args)
    if args.mode == 'test':
        Test.test(args)

if __name__ == '__main__':
    main()

from march_maddness_ai.predictor import Predictor
import march_maddness_ai.NeuralNetwork as NN
from bracket.bracket import Bracket
from march_maddness_ai.trainer import Trainer

from argparse import ArgumentParser
from typing import Sequence, Optional
from signal import signal, SIGTERM
import logging
import traceback

teams2025 : list[str] = [
    "St. John's", "Omaha",
    "Kansas", "Arkansas",
    "Texas Tech", "UNC Wilmington",
    "Missouri", "Drake",
    "Maryland", "Grand Canyon",
    "Memphis", "Colorado St.",
    "UConn", "Oklahoma",
    "Florida", "Norfolk St.",
    
    "Michigan St.", "Bryant",
    "Marquette", "New Mexico",
    "Iowa St.", "Lipscomb",
    "Ole Miss", "North Carolina",
    "Texas A&M", "Yale",
    "Michigan", "UC San Diego",
    "Louisville", "Creighton",
    "Auburn", "Alabama St.",
    
    "Duke", "Mount St. Mary's",
    "Mississippi St.", "Baylor",
    "Oregon", "Liberty",
    "Arizona", "Akron",
    "BYU", "VCU",
    "Wisconsin", "Montana",
    "Saint Mary's", "Vanderbilt",
    "Alabama", "Robert Morris",
    
    "Houston", "SIU Edwardsville",
    "Gonzaga", "Georgia",
    "Clemson", "McNeese",
    "Purdue", "High Point",
    "Illinois", "Xavier",
    "Kentucky", "Troy",
    "UCLA", "Utah St.",
    "Tennessee", "Wofford"
]
teams2026 : list[str] = [
    "Idaho", "Houston",
    "Saint Mary's", "Texas A&M",
    "Penn", "Illinois",
    "North Carolina", "VCU",
    "Nebraska", "Troy",
    "McNeese", "Vanderbilt",
    "Clemson", "Iowa",
    "Prairie View A&M", "Florida",
    
    "UConn", "Furman",
    "UCLA", "UCF",
    "North Dakota St.", "Michigan St.",
    "Louisville", "South Florida",
    "Cal Baptist", "Kansas",
    "St. John's", "Northern Iowa",
    "TCU", "Ohio St.",
    "Siena", "Duke",
    
    "Arizona", "Long Island",
    "Villanova", "Utah St.",
    "Wisconsin", "High Point",
    "Arkansas", "Hawai'i",
    "BYU", "Texas",
    "Gonzaga", "Kennesaw St.",
    "Miami FL", "Missouri",
    "Purdue", "Queens",
    
    "Michigan", "Howard",
    "Georgia", "Saint Louis",
    "Texas Tech", "Akron",
    "Alabama", "Hofstra",
    "Tennessee", "Miami OH",
    "Virginia", "Wright St.",
    "Kentucky", "Santa Clara",
    "Iowa St.", "Tennessee St."
]
    
def predict(args):
    print(args)
    teams : list[str] = []
    match args.year:
        case 2025:
            teams = teams2025
        case 2026:
            teams = teams2026
        case _:
            raise Exception(f"Year {args.year} is not a valid year")
        
    predictor : Predictor = Predictor(teams, args.year, args.filepath, args.layers)
    # predictor : Predictor = Predictor(teams, year, "trained_models\\2026\\NN256_15000.npy", [NN.Sigmoid(38,256), NN.Sigmoid(256,1)])
    results = predictor.get_results()

    bracket = Bracket(64, True, results)
    bracket.show()

def train(args):
    trainer = Trainer(args.epochs, args.layers, args.filepath, args.load)
    trainer.train()

def collect_data(args):
#     start_dates = [["2025", "2026", "1101", "0317"],
#         ["2024", "2025", "1101", "0318"],
#         ["2023", "2024", "1101", "0318"],
#         ["2022", "2023", "1101", "0313"],
#         ["2021", "2022", "1101", "0312"],
#         ["2020", "2021", "1101", "0317"],
#         ["2019", "2020", "1101", "0316"],
#         ["2018", "2019", "1101", "0321"],
#         ["2017", "2018", "1101", "0312"],
#         ["2016", "2017", "1101", "0316"],
#         ["2015", "2016", "1101", "0314"],
#         ["2014", "2015", "1101", "0316"],
#         ["2013", "2014", "1101", "0317"],
#         ["2012", "2013", "1101", "0318"],
#         ["2011", "2012", "1101", "0312"],
#         ["2010", "2011", "1101", "0314"],
#         ["2009", "2010", "1101", "0315"],
#         ["2008", "2009", "1101", "0316"],
#         ["2007", "2008", "1101", "0317"]]
    pass

def parse_args(argv: Optional[Sequence[str]] = None):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help='subcommand help', required=True)
    
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("-l", "--layers", help="The layers for the model", type=str, required=True)
    train_parser.add_argument("-e", "--epochs", help="The number of epochs", type=int, required=True)
    train_parser.add_argument("-f", "--filepath", help="The base path to save the model to. If not loading, does not want extension", required=True)
    train_parser.add_argument("--load", action="store_true", help="Defines whether to load the model from the supplied filepath.")
    train_parser.set_defaults(func=train)
    
    predict_parser = subparsers.add_parser("predict", help="Predict with the model")
    predict_parser.add_argument("-y", "--year", choices=[2025,2026], help="The year to predict on", type=int, required=True)
    predict_parser.add_argument("-f", "--filepath", help="The model filepath to use. note: requires .npy extension", required=True)
    predict_parser.add_argument("-l", "--layers", help="The layers for the model", type=str, required=True)
    predict_parser.set_defaults(func=predict)
    
    collect_parser = subparsers.add_parser("collect", help="Collect data")
    collect_parser.add_argument("-d", "--dates", type=list[list[str]], help="List of [start_year, end_year, start_MMDD, end_MMDD]", required=True)
    collect_parser.set_defaults(func=collect_data)

    return parser.parse_args(argv)

def sigterm_handler(_, __):
    raise SystemExit(1)

def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    if hasattr(args, "layers"):
        args.layers = eval(args.layers)
    if hasattr(args, "dates"):
        args.dates = eval(args.dates)
        
    args.func(args)

def __main__():
    signal(SIGTERM, sigterm_handler)

    try:
        main()
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        logging.error(traceback.format_exc())
        print(e)
        exit(1)

if __name__ == "__main__":
    __main__()
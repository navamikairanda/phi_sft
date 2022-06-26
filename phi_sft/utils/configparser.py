import configparser

def config_parser(args_file):	
	args = configparser.ConfigParser() 
	args.read_file(open(args_file))
	return args
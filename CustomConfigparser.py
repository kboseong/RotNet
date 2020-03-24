import configparser


def CustomConfigparser(field, config):
    F_config = configparser.ConfigParser(allow_no_value=True)
    T_config = configparser.ConfigParser(allow_no_value=True)
    F_config.read(field)
    T_config.read(config)
    sections = F_config.sections()
    r = {}
    for section in sections:
        items = F_config.items(section)
        sub_r = {}
        for ele in items:
            try :
                sub_r[ele[0]] = T_config[section][ele[0]]
            except :
                sub_r[ele[0]] = None
        r[section]=sub_r
    return r
    
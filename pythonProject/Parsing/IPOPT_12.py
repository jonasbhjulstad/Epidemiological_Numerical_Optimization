import pandas as pd


def find_line_index(table_title,data_list, start=0):
    for index, line in enumerate(data_list[start:]):
        if table_title in line:
            break
    return index + start

def filter_lines(line_str, data_list, start=0):
    for index, line in reversed(list(enumerate(data_list))):
        if index == start:
            break
        if line_str in line:
            del data_list[index]
    return data_list

def convert_to_rows(l,alpha_pr_col):
    number_columns = 10

    tags = ['f', 'F', 'h','H','k','K','n','N','R','w','s','S','t','T','r']

    l = l.replace('r',' ')
    for i in tags:
        l = l.replace(i, '')

    split_data = l.split(' ')

    for index, elem in reversed(list(enumerate(split_data))):
        if elem == '':
            del split_data[index]
        if elem == '-':
            split_data[index] = -1.0
        else:
            split_data[index] = float(split_data[index])
    return split_data

def take_table(txt_table):
    iter = []
    objective = []
    inf_pr = []
    inf_du = []
    lg_mu = []
    abs_d = []
    lg_rg = []
    alpha_du = []
    alpha_pr = []
    ls = []

    alpha_pr_ind = 8

    for index, r in enumerate(txt_table):
        data_row = convert_to_rows(r, alpha_pr_ind)
        if data_row:
            cols= data_row
            iter.append(cols[0])
            objective.append(cols[1])
            inf_pr.append(cols[2])
            inf_du.append(cols[3])
            lg_mu.append(cols[4])
            abs_d.append(cols[5])
            lg_rg.append(cols[6])
            alpha_du.append(cols[7])
            alpha_pr.append(cols[8])
            ls.append(cols[9])

    table_data = {'iter': iter, 'objective': objective,
                  'inf_pr': inf_pr, 'inf_du': inf_du,
                  'lg_mu': lg_mu, 'abs_d': abs_d,
                    'lg_rg': lg_rg, 'alpha_du': alpha_du,
                    'alpha_pr': alpha_pr, 'ls': ls}
    return table_data

def parse_IPOPT_log(file, output_file):
    content = []
    table_title = 'c scaling vector'
    table_end = 'Number of Iterations....:'
    line1 = 'DenseVector'
    line2 = 'Homogeneous'
    slack_line = 'Slack too small'
    with open(file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    table_start = find_line_index(table_title, content)
    content = filter_lines(table_title,content, start=table_start+1)
    content = filter_lines(line1,content, start=table_start+1)
    content = filter_lines(line2,content, start=table_start+1)
    content = filter_lines(slack_line, content)
    table_data = content[table_start+1:]
    table = take_table(table_data)

    df = pd.DataFrame(table).set_index('iter')

    df.to_pickle(output_file)
    return df

if __name__ == '__main__':
    df = parse_IPOPT_log(r'../data/log.opt', r'../data/parsed.pck')

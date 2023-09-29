from datetime import datetime
import json
import os
import re

def check_and_mkdir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('make dir:', dirname)
    else:
        print(dirname, 'exists, no change made')

def str_to_datetime(s):
    # input: e.g. '2016-10-14'
    # output: datetime.datetime(2016, 10, 14, 0, 0)
    # Other choices:
    #       pd.to_datetime('2016-10-14')  # very very slow
    #       datetime.strptime('2016-10-14', '%Y-%m-%d')   #  slow
    # ymd = list(map(int, s.split('-')))
    ymd = list(map(int, re.split(r'[-\/:.]', s)))
    assert (len(ymd) == 3) or (len(ymd) == 1)
    if len(ymd) == 3:
        assert 1 <= ymd[1] <= 12
        assert 1 <= ymd[2] <= 31
    elif len(ymd) == 1:
        ymd = ymd + [1, 1]  # If only year, set to Year-Jan.-1st
    return datetime(*ymd)

def is_mci(code):
    """
    ICD9:
        331.83    Mild cognitive impairment, so stated
        294.9     Unspecified persistent mental disorders due to conditions classified elsewhere
    ICD10:
        G31.84    Mild cognitive impairment, so stated
        F09      Unspecified mental disorder due to known physiological condition
    :param code: code string to test
    :return: true or false
    """
    assert isinstance(code,str)
    code_set = ('331.83','294.9','G31.84', 'F09','33183','2949','G3184')
    return code.startswith(code_set)

def is_AD(code):
    """
    ICD9:
        331.0  Alzheimer's disease
    ICD10:
        G30:  Alzheimer's disease
        G30.0  Alzheimer's disease  with early onset
        G30.1  Alzheimer's disease  with late onset
        G30.8  other Alzheimer's disease
        G30.9  Alzheimer's disease, unspecified


    :param code: code string to test
    :return: true or false
    """
    assert isinstance(code,str)
    code_set = ('331.0','3310','G30')
    return code.startswith(code_set)

def is_PD(code):
    """
    ICD9:
        332  Parkinson's disease
    ICD10:
        G20: Parkinson's disease


    :param code: code string to test
    :return: true or false
    """
    assert isinstance(code,str)
    code_set = ('332','G20')
    #code_set = ('G20', '332', '332.0', '332.00', '3320') # jie xu code list --- no change in INSIGHT cohort
    return code.startswith(code_set)

def is_PDRD(code):
    # PD related dx
    """
    ICD9:
        332.0  Paralysis agitans
        332.1  Secondary parkinsonism
    ICD10:
        G21: Secondary parkinsonism
        G21.1:
        G21.2:
        G21.3:
        G21.8:
        G21.9:
        G23:
        G23.1:
        G23.9:



    :param code: code string to test
    :return: true or false
    """
    assert isinstance(code,str)
    code_set = ('332.0','332.1','G21','G21.1','G21.2','G21.3','G21.8','G21.9','G23','G23.1','G23.9')
    # code_set = ('G21.4', 'G21.9', 'G23.1', 'G21.8', 'G23.9', 'G21.2', 'G23.0', 'G21.19', 'G21.3', 'G21.11', 'G21.0', '332.1',
    # 'G2111', 'G231', 'G219', '3321', 'G210', 'G218', 'G2119', 'G239') # jie xu code list
    return code.startswith(code_set)

def is_CI(code):
    """
    ICD9:  '331.0',
           '290.12', '290.40', '290.41', '290.42', '290.43', '294.10', '294.11', '294.20', '294.21', '331.11', '331.19', '331.82',
           '331.83', '294.9'

    ICD10:
        'G30', 'G30.0', 'G30.1', 'G30.8', 'G30.9',
        'F02.80', 'F02.81', 'F02.90', 'F03.91', 'F04', 'G31.01', 'G31.09', 'G31.83', 'F01.50', 'F01.51',
        'G31.84', 'F09',
        'R41.81'


    :param code: code string to test
    :return: true or false
    """
    assert isinstance(code,str)
    code_set = ('331.0', '290.12', '290.40', '290.41', '290.42', '290.43', '294.10', '294.11', '294.20', '294.21', '331.11', '331.19', '331.82','331.83', '294.9')
    code_set += ('G30', 'G30.0', 'G30.1', 'G30.8', 'G30.9',
        'F02.80', 'F02.81', 'F02.90', 'F03.91', 'F04', 'G31.01', 'G31.09', 'G31.83', 'F01.50', 'F01.51',
        'G31.84', 'F09',
        'R41.81')
    return code.startswith(code_set)

def is_dementia(code):
    """
    ICD9:
       '294.10','294.11','294.20','294.21'
       '290.-'  all codes and variants in 290.- tree

    ICD10:
        'F01.-'  all codes and variants in this tree
        'F02.-'  all codes and variants in this tree
        'F03.-'  all codes and variants in this tree

    :param code: code string to test
    :return: true or false
    """
    assert isinstance(code,str)
    code_set = ('294.10','294.11','294.20','294.21','2941','29411','2942','29421')
    code_set += ('290',)
    code_set += ('F01','F02','F03')
    return code.startswith(code_set)

def is_ND(code, ND_stand_code):
    """
        # ND_stand_code['fall'] = fall_code
        # ND_stand_code['dementia'] = dementia_code
        # ND_stand_code['mental'] = mental_code
        # ND_stand_code['PIGD'] = PIGD_code
        # ND_stand_code['CI'] = CI_code

    :param code: code string to test
    :return: true or false
    """
    assert isinstance(code,str)
    # code_set = ND_stand_code['fall']
    code_set = ND_stand_code['dementia']

    # code_set += ND_stand_code['mental']
    # code_set += ND_stand_code['PIGD']
    # code_set += ND_stand_code['CI']
    return code.startswith(code_set)

def load_model(model_class, filename):
    def _map_location(storage, loc):
        return storage

    # load trained on GPU models to CPU
    map_location = None
    if not torch.cuda.is_available():
        map_location = _map_location

    state = torch.load(str(filename), map_location=map_location)

    model = model_class(**state['model_params'])
    model.load_state_dict(state['model_state'])

    return model


def save_model(model, filename, model_params=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    state = {
        'model_params': model_params or {},
        'model_state': model.state_dict(),
    }
    check_and_mkdir(str(filename))
    torch.save(state, str(filename))



def load_icd_to_ccw(path):
    """ e.g. there exisit E93.50 v.s. E935.0.
            thus 5 more keys in the dot-ICD than nodot-ICD keys
            [('E9350', 2),
             ('E9351', 2),
             ('E8500', 2),
             ('E8502', 2),
             ('E8501', 2),
             ('31222', 1),
             ('31200', 1),
             ('3124', 1),
             ('F919', 1),
             ('31281', 1)]
    :param path: e.g. 'mapping/CCW_to_use_enriched.json'
    :return:
    """
    with open(path) as f:
        data = json.load(f)
        print('len(ccw_codes):', len(data))
        name_id = {x: str(i) for i, x in enumerate(data.keys())}
        id_name = {v:k for k, v in name_id.items()}
        n_dx = 0
        icd_ccwid = {}
        icd_ccwname = {}
        icddot_ccwid = {}
        for name, dx in data.items():
            n_dx += len(dx)
            for icd in dx:
                icd_no_dot = icd.strip().replace('.', '')
                if icd_no_dot:  # != ''
                    icd_ccwid[icd_no_dot] = name_id.get(name)
                    icd_ccwname[icd_no_dot] = name
                    icddot_ccwid[icd] = name_id.get(name)
        data_info = (name_id, id_name, data)
        return icd_ccwid, icd_ccwname, data_info

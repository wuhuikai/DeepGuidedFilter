from utils import task_name
from test_base import compare

TASK = task_name()
GT_ROOT  = '../hdrnet/dataset/{}/test/output'.format(TASK)
PRE_ROOT = 'results/{}/HDR_NET/PRE'.format(TASK)

compare(PRE_ROOT, GT_ROOT)
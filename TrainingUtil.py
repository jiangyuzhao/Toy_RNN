import inspect
from tensorflow.contrib.rnn import DropoutWrapper
from pydoc import locate


def get_rnn_cell(cell_class,
                 cell_params,
                 dropout_input_keep_prob: float = 1.0,
                 dropout_output_keep_prob: float = 1.0
                 ):
    cell_class = locate(cell_class)

    # Make sure additional arguments are valid
    cell_args = set(inspect.getfullargspec(cell_class.__init__).args[1:])
    for key in cell_params.keys():
        if key not in cell_args:
            raise ValueError(
                """{} is not a valid argument for {} class. Available arguments
                are: {}""".format(key, cell_class.__name__, cell_args))

    cell = cell_class(**cell_params)
    if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
        cell = DropoutWrapper(
            cell=cell,
            input_keep_prob=dropout_input_keep_prob,
            output_keep_prob=dropout_output_keep_prob
        )
    return cell

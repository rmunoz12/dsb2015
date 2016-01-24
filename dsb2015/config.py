class Config(object):
    """
    Stores path options. Note that all path attributes are expected to end with
    a slash `/`.
    """

    out_files = \
        {'train_data_csv': '../output/train-64x64-data.csv',
         'train_label_csv': '../output/train-label.csv',
         'train_systole_out': '../output/train-stytole.csv',
         'train_diastole_out': '../output/train-diastole.csv',
         'valid_data_csv': '../output/validate-64x64-data.csv',
         'valid_label_out': '../output/validate-label.csv',
         'sample_submit': '../data/sample_submission_validate.csv',
         'submit_out': '../output/submission.csv',
         'model_out': '',       # TODO save model output
         'test_data_path': ''}  # TODO load test_data path (round 2)

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        for name in Config.out_files:
            if self.local and name in {'train_data_csv', 'train_label_csv',
                                       'valid_data_csv', 'valid_label_out'}:
                setattr(self, name, self.output_path + 'local_' + name)
            else:
                setattr(self, name, self.output_path + name)

    def __repr__(self):
        return str(self.__dict__)

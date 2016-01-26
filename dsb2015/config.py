class Config(object):
    """
    Stores path options. Note that all path attributes are expected to end with
    a slash `/`.
    """

    out_files = \
        {'train_data_csv': 'train-64x64-data.csv',
         'train_label_csv': 'train-label.csv',
         'train_systole_out': 'train-stytole.csv',
         'train_diastole_out': 'train-diastole.csv',
         'valid_data_csv': 'validate-64x64-data.csv',
         'valid_label_out': 'validate-label.csv',
         'sample_submit': 'sample_submission_validate.csv',
         'submit_out': 'submission.csv',
         'model_out': '',       # TODO save model output
         'test_data_path': ''}  # TODO load test_data path (round 2)

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        for k, v in Config.out_files.items():
            if self.local and k in {'train_data_csv', 'train_label_csv',
                                       'valid_data_csv', 'valid_label_out'}:
                setattr(self, k, self.output_path + 'local_' + v)
            elif k in {'sample_submit'}:
                setattr(self, k, self.train_path + v)
            else:
                setattr(self, k, self.output_path + v)

    def __repr__(self):
        return str(self.__dict__)

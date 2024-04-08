class Config:
    def __init__(
        self,
        method,
        num_points,
        num_class,
        input_dim,
        init_hidden_dim,
        k,
        tome_further_ds=None,
        tome_further_ds_use_xyz=False,
        **kwargs
    ):
        self.method = method
        self.num_points = num_points
        self.num_class = num_class
        self.input_dim = input_dim
        self.init_hidden_dim = init_hidden_dim
        self.k = k
        self.tome_further_ds = tome_further_ds
        self.tome_further_ds_use_xyz = tome_further_ds_use_xyz
        for name, val in kwargs.items():
            setattr(self, name, val)

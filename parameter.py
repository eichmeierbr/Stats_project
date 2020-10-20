#!/usr/bin/env python


class Parameter:
    def __init__(self, values, value, lin=True, categ=False):
        self.options = values
        self.value = value
        self.linear = lin # Linear vs. Lograithmic scaling
        self.categorical = categ

  
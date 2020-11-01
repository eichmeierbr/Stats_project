#!/usr/bin/env python


class Parameter:
    def __init__(self, values, count=0, lin=True, categ=False):
        self.options = values
        self.value = values[0]
        self.linear = lin # Linear vs. Lograithmic scaling
        self.samples = 0
        if count==0:
            self.samples = len(values)
        else:
            self.samples = count
        self.categorical = categ

  
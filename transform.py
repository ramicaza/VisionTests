class Transform:
    def __init__(self,transform_coefficient):
        self.transform_coefficient = transform_coefficient

    def transform(self,x,y):
        factor = self.transform_coefficient
        return (x/factor,y/factor)

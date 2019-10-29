class FeatureTargetData:
    """Data structure to hold feature,target and type of learning like regression
    """

    def __init__(self,X,y,type_cf):
        """Constructor
        
        Arguments:
            X {pandas.DataFrame} -- features
            y {pandas.DataFrame} -- target
            type_cf {string} -- regression or classification
        """
        self.X = X
        self.y = y
        self.type = type_cf

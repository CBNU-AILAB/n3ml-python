class Bohte:
    def __init__(self):
        pass

    def step(self, model, spiked_label):
        print(model)
        for l in reversed(model.layer):
            print(l)
        for l in reversed(model.layer):
            if l is next(reversed(model.layer)):
                print("last layer")
            else:
                print("hidden layer")

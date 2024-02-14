from datetime import date, datetime, timedelta

def Time_Costing(func):
    def core(*args, **kwargs):
        print('------------------------------------------------------------------------')
        start = datetime.now()
        print(start, 'Start Function')
        
        result = func(*args, **kwargs)
        
        print(datetime.now(), 'End Function, Time costing:', datetime.now() - start)
        print('------------------------------------------------------------------------')
        return result
    return core
from ..relevec import ReleVec

# relevec % python3 -m rvp.test.t-01 


# Create test classes and instances
#Cn = ReleVec.get_specified_subclass('Cn', dim_names=['a','b','c'])

ReleVec.import_subclasses_dict({
    'Ci': {'dim_ct': 3},
    'Cn': {'dim_names': ['a', 'b', 'c']}
    })
Cn = ReleVec.get_subclass_by_name('Cn')
on = Cn({'a':0.01, 'b':0.11, 'c':0.21})

print('Export vec instance:')

print(on.export_dict())

print('')

print("Export subs: ", ReleVec.export_subclasses_dict())

print("t-01.py test complete")
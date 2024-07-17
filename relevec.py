import sys
import typing
import math


#=====================#
#  custom exceptions  #
#=====================#

class ReleVecCustomException(Exception):
    pass


#===================#
#  class ReleVec  #
#===================#

class ReleVec:
    _eps = 0.000001   # A chosen epsilon value for float compare (vec mag etc.).
    _exportable_subclasses = {}   # Key is class name, value is the created subclass.


    @classmethod
    def equality_threshold(cls, val1:float, val2:float):
        return abs(val1 - val2) < MatchVecBase._eps


    @classmethod
    def is_valid_dimidx(cls, dimidx:int) -> bool:
        assert isinstance(dimidx, int), "ERROR: dimidx must be an int."
        return 0 <= dimidx    # Only restriction is non-negative int.


    def get_dimval_by_dimidx(self, dimidx: int) -> float:
        assert self.is_valid_dimidx(dimidx), ("The dimidx parameter "
            + f"(value: {dimidx}) is invalid.")
        try:
            return self.sparse_vec[dimidx]
        except KeyError:
            # By definition, value of zero is implied for any omitted dimidx of sparse_vec
            return 0.00


    def set_dimval_by_dimidx(self, dimidx:int, dimval:float):
        assert self.is_valid_dimidx(dimidx), ("The dimidx parameter "
            + f"(value: {dimidx}) is invalid.")
        assert isinstance(dimval, float), "ERROR: dimval must be a float."
        self.sparse_vec[dimidx] = dimval
       

    def set_vec_by_dim_tuples(self, dim_tuples:tuple[int, float], strict=False):
        assert not strict, ("The strict parameter does not yet support the value True. "
            + "When this feature is added, it will enforce uniqueness among the first "
            + "element (ie. dimension key) of each member of dim_tuples.")
        self.sparse_vec = {}
        for dim_tuple in dim_tuples:
            assert len(dim_tuple) == 2, ("The dim_tuples parameter contains a tuple of "
                + "invalid size. Only tuples of two elements serving as key and value "
                + "are allowed.")
            self.set_dimval_by_dimidx(dim_tuple[0], dim_tuple[1])
    
    
    def set_vec_by_dimdict(self, dimdict:dict):
        # Since dimdict keys are unique, no need for strict with set_vec_by_dim_tuples()
        self.set_vec_by_dim_tuples(list(dimdict.items()), False)


    def get_magnitude_squared(self):
        return sum([dimval**2 for dimval in self.sparse_vec.values()])


    def get_magnitude(self):
        return math.sqrt(self.get_magnitude_squared())


    def normalize(self):
        magnitude_squared = self.get_magnitude_squared()
        if magnitude_squared < MatchVecBase._eps:
            raise ValueError("Cannot normalize a vector with zero magnitude.")
        elif equality_threshold(magnitude_squared, 1.00):
            pass
        else:
            magnitude = math.sqrt(magnitude_squared)
            for dimidx, dimval in self.sparse_vec.items():
                self.sparse_vec[dimidx] = dimval / magnitude    


    def dot_product(self, target_vec):
        class_name = type(self).__name__
        assert isinstance(target_vec, type(self)), ("ERROR: "
            + f"target_vec is not an instance of {class_name}")
        dot_product = 0.0
        for dimidx, dimval in self.sparse_vec.items():
            dot_product += dimval * target_vec.get_dimval_by_dimidx(dimidx)
        return dot_product


    def __init__(self, dimdict = None):
        if dimdict is None:
            # Initialize empty sparse_vec dict
            self.sparse_vec = {}
        else:
            self.set_vec_by_dimdict(dimdict)


    #--------------------------#
    #  ReleVec enhancements  #
    #--------------------------#

    def is_normalized(self):
        return equality_threshold(self.get_magnitude_squared(), 1.00)


    #-------------------------------------#
    #  custom exceptions as inner classes #
    #-------------------------------------#

    class SvInstanceDictError(              ReleVecCustomException ):
        pass


    class SvSubclassNameConventionError(    ReleVecCustomException ):
        pass


    class SvSubclassNameRedefinitionError(  ReleVecCustomException ):
        pass


    class SvSubclassNameInUse(              SvSubclassNameRedefinitionError ):
        pass


    class SvNonSubclassGlobalNameInUse(     SvSubclassNameRedefinitionError ):
        pass


    #--------------------#
    #  Input Validation  #
    #--------------------#

    @staticmethod
    def reject_bad_subclass_name( name:str ):
        # Return None if name is accepted, otherwise string stating reason for rejection.
        if not isinstance(name, str):
            raise SvSubclassNameConventionError("The name is not a str.")
        elif len(name) < 1:
            raise SvSubclassNameConventionError("The name is an empty str.")
        elif name[0].islower():
            raise SvSubclassNameConventionError("The name begins with a lower case "
                + "letter but uppercase is expected.")
        elif name[0].isdigit():
            raise SvSubclassNameConventionError("The name begins with a digit.")
        elif name[0].isupper():
            pass
        elif name[0] == '_' and len(name) > 1:
            # The name can begin with underscore as long as the second letter is uppercase.
            if name[1].isupper():
                pass
            else:
                raise SvSubclassNameConventionError("Since the name begins with an "
                    + "underscore, the second letter was expected to be uppercase but "
                    + "was not.")
        else:
            raise SvSubclassNameConventionError("The name does not begin with uppercase "
                + " or underscore as expected.")
        
        
    @staticmethod
    def reject_unavailable_subclass_name( name:str ):
        if name in ReleVec._exportable_subclasses:
            raise SvSubclassNameInUse("The name is already used by a subclass of "
                + "ReleVec." )
        elif name in globals():
            raise SvNonSubclassGlobalNameInUse("The name is already used by a global "
                + "not created via ReleVec.get_specified_subclass().")
        

    @staticmethod
    def reject_bad_instance_dict( instance_dict:dict ):
        #
        if not isinstance(instance_dict, dict):
            raise SvInstanceDictError("Expected dict input but instead received "
                + f"{type(instance_dict).__name__} consisting of {instance_dict}")
        #
        class_name = instance_dict.get('class_name')
        ReleVec.reject_bad_subclass_name(class_name)
        #
        dimdict = instance_dict.get('dimdict')
        if not isinstance(class_name, dict):
            raise SvInstanceDictError("Expected dict input but instead received "
                + f"{type(instance_dict).__name__} consisting of {dimdict}")
        
        
    #---------------------#
    #  fetch a subclass   #
    #---------------------#

    @staticmethod
    def get_subclass_by_name(
        class_name:str = "ReleVec"
    ) -> type:
        a_class = None
        #
        try:
            a_class = ReleVec._exportable_subclasses[class_name]
        except KeyError:
            if "ReleVec" == class_name:
                # Just return the ReleVec class itself.
                a_class = ReleVec
        #
        if a_class is None:
            pass
            # ReleVec.reject_bad_subclass_name(class_name)
        else:
            assert issubclass(a_class, ReleVec), (f"The requested name {class_name} "
                + "was registered with ReleVec._exportable_subclasses but "
                + "unexpectedly not found to be a subclass of ReleVec as required.")
        #
        return a_class


    # get_specified_subclass()      -- To be monkeypatched.
    
    
    #--------------------#
    #   Serialization    #
    #  of instance data  #
    #--------------------#

    def export_dict(self):
        # Starting with empty list, accumulate labeled dimension tuples.
        dimdict = {}
        for dimidx, dimval in self.sparse_vec.items():
            dimdict[dimidx] = dimval
        return {'class_name': type(self).__name__,
            'dimdict':dimdict}


    def import_dict(self, serialized_instance_dict:dict, strict=False):
        ReleVec.reject_bad_instance_dict(serialized_instance_dict)
        self.set_vec_by_dimdict(serialized_instance_dict['dimdict'])
     

    def __repr__(self):
        return ("an instance of class " + self.__class__.__name__
            + " containing " + str(self.export_dict()) )


    @classmethod
    def export_substruct_dict(cls):
        # Return dict specifying class attributes beyond the base class so that the class
        # can be reconstructed from serialized data in any program using this module.
        return {}   # The base class declares nothing beyond itself.
    
    
    # import_substruct_dict()           -- To be monkeypatched.
    
    
    @staticmethod
    def export_subclasses_dict():
        # Iterate through ReleVec._exportable_subclasses.
        all_subs = {}
        for sub_name, sub_class in ReleVec._exportable_subclasses.items():
            all_subs[sub_name] = sub_class.export_substruct_dict()
            #print(f"sub {sub_name}: ", sub_class)
        return all_subs

            
    # import_subclasses_dict()      -- To be monkeypatched.
    
    
    #------------------#
    #  Monkeypatching  #
    #------------------#

    # NOTE -- Methods to be monkeypatched:
    #
    # ReleVec.get_specified_subclass()  -- a staticmethod
    # ReleVec.import_substruct_dict()   -- a staticmethod
    # ReleVec.import_subclasses_dict()  -- a staticmethod
    #
    # The above method(s) will be monkeypatched into class ReleVec after some internal
    # definitions they depend on are defined. In some cases, these internal definitions
    # could not be made until a valide definition of the core portion of ReleVec
    # has been completed.


#=====================================#
#  internal class _SV_LimitedIdxBase  #
#=====================================#

class _SV_LimitedIdxBase(ReleVec):
    _dimidx_ct = 0        # Let ReleVec.get_specified_subclass() override in a subclass.


    @classmethod
    def is_valid_dimidx(cls, dimidx:int) -> bool:
        assert isinstance(dimidx, int), "ERROR: dimidx must be an int."
        return 0 <= dimidx and dimidx < cls._dimidx_ct


    def __init__(self,
        dimdict = None,
        required_direct_parent_name = "_SV_LimitedIdxBase"
        # Would prefer _SV_LimitedIdxBase.__name__ over literal but class not yet defined.
    ):
        required_direct_parent_found = False
        for parent in type(self).__bases__:
            # print("Checking parent name: %s" % parent.__name__)
            if required_direct_parent_name == parent.__name__:
                required_direct_parent_found = True
                break
        assert required_direct_parent_found, ("An attempt was made to create an instance "
            + "of an internal subclass of ReleVec. This is not supported. Please use"
            + "ReleVec.create_subclass() to create any subclasses of ReleVec.")
        super().__init__(dimdict = dimdict)


    @classmethod
    def export_substruct_dict(cls) -> dict:
        # Return dict specifying class attributes beyond the base class so that the class
        # can be reconstructed from serialized data in any program using this module.
        return {"dim_ct": cls._dimidx_ct}


    # import_substruct_dict()         -- Inherit from ReleVec classmethod monkeypatch.


#===================================#
#  internal class _SV_NamedDimBase  #
#===================================#

class _SV_NamedDimBase(_SV_LimitedIdxBase):
    _dim_registry = {}    # Let ReleVec.get_specified_subclass() override in a subclass.


    def get_dimval_by_dimnam(self, dimnam:str):
        assert isinstance(dimnam, str), f"Parameter dimnam (value: {dimnam}) is not str."
        try:
            dimidx = self._dim_registry[dimnam]
        except KeyError:
            raise ValueError(f"Parameter dimnam (value: {dimnam}) was not declared.")
        return self.get_dimval_by_dimidx(dimidx)


    def set_dimval_by_dimnam(self, dimnam:str, dimval:float):
        assert isinstance(dimnam, str), f"Parameter dimnam (value: {dimnam}) is not str."
        try:
            dimidx = self._dim_registry[dimnam]
            # Use dimidx to set dimval
            self.set_dimval_by_dimidx(dimidx, dimval)
        except KeyError:
            raise ValueError(f"Parameter dimnam (value: {dimnam}) was not declared.")
    
    
    def set_vec_by_dim_tuples(self, dim_tuples:tuple[typing.Any, float], strict=False):
        assert not strict, ("The strict parameter does not yet support the value True. "
            + "When this feature is added, it will enforce uniqueness among the first "
            + "element (ie. dimension key) of each member of dim_tuples.")
        if len(dim_tuples) > 0:
            # Treat the first element of each tuple of dim_tuples as a dimension key.
            # They correspond dimdict keys to when called from set_vec_by_dimdict().
            # Using the key from the first tuple as a model, enforce that all
            # keys are of same type which must be either int or str.
            if isinstance(dim_tuples[0][0], str):
                # Expect all dimension keys to be names since the first one was a str.            
                self.sparse_vec = {}   # Clear vector before setting new data.
                for dim_tuple in dim_tuples:
                    assert len(dim_tuple) == 2, ("The dim_tuples parameter contains a "
                        + "tuple of invalid size. Only tuples of two elements serving "
                        + "as key and value are allowed.")
                    assert isinstance(dim_tuple[0], str), ("Since the first member of the "
                        + "first tuple was of type str, all must be. "
                        + f"However, member {dim_tuple} does not qualify.")
                    self.set_dimval_by_dimnam(dim_tuple[0], dim_tuple[1])
            else:
                # Expect all dimension keys to be dimidx since the first one was an int.
                super().set_vec_by_dim_tuples(dim_tuples, strict)            
        else:
            # len(dim_tuples) == 0
            self.sparse_vec = {}
            return


    def __init__(self,
        dimdict = None,
        required_direct_parent_name = "_SV_NamedDimBase"
        # Would prefer _SV_NamedDimBase.__name__ over literal but class not yet defined.
    ):
        # Same as super except one of the default params starts with a different value.
        super().__init__(
            dimdict = dimdict,
            required_direct_parent_name = required_direct_parent_name)


    def export_dict(self):
        # Starting with empty list, accumulate labeled dimension tuples.
        dimnam_list = list(self._dim_registry.items())
        dimdict = {}
        
        # print("Instance of " + type(self).__name__ + " method export_dict():")
        # print("dimnam_list: ", dimnam_list)
        
        for dimidx, dimval in self.sparse_vec.items():
            # dimnam_list[dimidx] yeilds tuple of dimnam, dimidx
            dimnam = dimnam_list[dimidx][0]   # dimnam comes first in tuple.
            assert dimnam_list[dimidx][1] == dimidx, ("Dim registry was not sorted "
                + "as expected. Perhaps dict insertion order was not preserved. "
                + "Module initialization code attempts to require a version of "
                + "Python (such as 3.7+) that preserves dict insertion order. See "
                + "https://mail.python.org/pipermail/python-dev/2017-December/151283.html")
            dimdict[dimnam] = dimval
        return {'class_name': type(self).__name__,
            'dimdict':dimdict}

    
    # import_dict()
    # The inherited import_dict() should work fine here. No need to override.


    @classmethod
    def export_substruct_dict(cls) -> dict:
        # Return dict specifying class attributes beyond the base class so that the class
        # can be reconstructed from serialized data in any program using this module.
        return {"dim_names": list(cls._dim_registry.keys())}


    # import_substruct_dict()         -- Inherit from ReleVec classmethod monkeypatch.
    

#=========================#
#  module monkeypatching  #
#=========================#

def _perform_module_monkeypatching():

    # ReleVec.get_specified_subclass() definition:
    @staticmethod
    def _get_specified_subclass_monkeypatch(
        class_name:str = None,
        dim_names:typing.Iterable[str] = None,  # previously named dimnam_iterable
        dim_ct:int = None                       # previously named limited_idx_ct
    ) -> type:
        # Attempt to find existing subclass named class_name or None if name is undefined.
        # TypeError exception raised if class_name is already defined but not a subclass.
        a_class = ReleVec.get_subclass_by_name(class_name)
        #
        ReleVec.reject_bad_subclass_name(class_name)
        # 
        if None == a_class:
            # Attempt to create a new subclass.
            if dim_names != None:
                dim_reg = {}
                for dimidx, dimnam in enumerate(dim_names):
                    assert isinstance(dimnam, str), ("Found a non str member of "
                        + "the dim_names parameter.")
                    assert dimnam not in dim_reg, (f"The name {dimnam} appeared "
                        + "multiple times in the dim_names parameter.")
                    dim_reg[dimnam] = dimidx
                # Create a subclass of _SV_NamedDimBase
                a_class = type(class_name, (_SV_NamedDimBase,), {
                    "_dimidx_ct": len(dim_reg),
                    "_dim_registry": dim_reg
                })
            elif dim_ct != None:
                # Create a subclass of _SV_LimitedIdxBase
                a_class = type(class_name, (_SV_LimitedIdxBase,), {
                    "_dimidx_ct": dim_ct
                })
            else:
                # Create an unembellished subclass of ReleVec
                a_class = type(class_name, (ReleVec,), {})
            # Confirm and register new exportable subclass.
            if None == a_class:
                raise RuntimeError("Unexpected failure to create a subclass "
                    + "of ReleVec.")
            else:
                ReleVec._exportable_subclasses[class_name] = a_class
        # Return a valid new or existing class unless an exception has been raised.
        return a_class
    

    #---------------------#
    #  New Serialization  #
    #---------------------#
    
    # ReleVec.import_substruct_dict() definition:
    @staticmethod
    def _import_substruct_dict_monkeypatch(
        class_name:str,
        substruct_dict:dict
        # , reject_redundant:bool = True
    ) -> type:
        a_class = None
        #
        ReleVec.reject_unavailable_subclass_name(class_name)
        ReleVec.reject_bad_subclass_name(class_name)
        #
        assert isinstance(substruct_dict, dict), ("The substruct_dict parameter was "
            + "not a dict as expected.")
        #
        dim_names = substruct_dict.get('dim_names')
        dim_ct = substruct_dict.get('dim_ct')
        #
        if dim_names != None:
            dim_reg = {}
            for dimidx, dimnam in enumerate(dim_names):
                assert isinstance(dimnam, str), ("Found a non str member of "
                    + "the dim_names parameter.")
                assert dimnam not in dim_reg, (f"The name {dimnam} appeared "
                    + "multiple times in the dim_names parameter.")
                dim_reg[dimnam] = dimidx
            # Create a subclass of _SV_NamedDimBase
            a_class = type(class_name, (_SV_NamedDimBase,), {
                "_dimidx_ct": len(dim_reg),
                "_dim_registry": dim_reg
            })
        elif dim_ct != None:
            # Create a subclass of _SV_LimitedIdxBase
            a_class = type(class_name, (_SV_LimitedIdxBase,), {
                "_dimidx_ct": dim_ct
            })
        else:
            # Create an unembellished subclass of ReleVec
            a_class = type(class_name, (ReleVec,), {})
        # Confirm and register new exportable subclass.
        if None == a_class:
            raise RuntimeError("Unexpected failure to create a subclass of ReleVec.")
        else:
            ReleVec._exportable_subclasses[class_name] = a_class
        # Return a valid new or existing class unless an exception has been raised.
        return a_class


    # ReleVec.import_subclasses_dict() definition:
    #   input sample: {'Ci': {'dim_ct': 2}, 'Cn': {'dim_names': ['a', 'b']}}
    @staticmethod
    def _import_subclasses_dict_monkeypatch(subclasses_dict:dict) -> type:
        #??? subtuples = dimnam_list = list(self._dim_registry.items())
        for subtuple in subclasses_dict.items():
            ReleVec.import_substruct_dict(subtuple[0], subtuple[1])


    #---------------------------------------------------#
    #  Body of function _perform_module_monkeypatching  #
    #---------------------------------------------------#

    ReleVec.get_specified_subclass  = _get_specified_subclass_monkeypatch
    ReleVec.import_substruct_dict   = _import_substruct_dict_monkeypatch
    ReleVec.import_subclasses_dict  = _import_subclasses_dict_monkeypatch


#=========================#
#  Module Initialization  #
#=========================#

min_req_py_ver = (3, 12)    # Required version >= Python 3.12
# print("Loading relevec.py module.")
assert sys.version_info >= min_req_py_ver, "{pre} {cur_V} {but} {req_v}.".format(
    pre   = "The currently executing Python version is",
    cur_V = "%d.%d" % (sys.version_info[0], sys.version_info[1]),
    but   = "but this module requires at least",
    req_v = "%d.%d" % (min_req_py_ver[0], min_req_py_ver[1]) )
_perform_module_monkeypatching()
print("Module relevec.py loaded.")
    

#=================#
#  Documentation  #
#=================#

# Abbreviations:
#   Vec -- Vector
#   Dim -- Dimension
#   Idx -- Index
#   Nam -- Name
#   Val -- Value
#   Ct  -- Count
#
# Acronyms
#   dimdict -- a dictionary of dimension keys and values -- keys may be str or int
#   dimidx -- an int index that specifies a dimension
#   dimnam -- a str name that specifies a dimension
#   dimval -- a float specifying the magnitude of a vector in a certain dimension
#   dimidx_ct -- a count specifying a zero based range of dimidx values
import time
import numbers
import copy

def main():
    # Real vectors (are matrices)
    v = Matrix.vector_from_list([1, 2, 3])
    # print(v)

    # Complex vectors
    u = Matrix.vector_from_list([4+1j, 5-1j, 6])
    # print(u)

    # Real matrices
    m = Matrix([[1, 2],[3, 4]])
    # print(m)

    # Complex matrices
    n = Matrix([[1+2j, 3+4j], [5+6j, 7+8j]])
    # print(n)

    # Nice subscripting, including submatrices
    # print(m[0,0])
    # print(m[0,:])
    # print(m[:,0])
    m1 = m.copy()
    m1[0,0] = -1
    # print(m1)
    m1[0, :] = Matrix([[-2, -3]])
    # print(m1)
    m1[:, 0] = [[-4],[-5]]
    # print(m1)

    # Transposition
    a = Matrix([[1,2,3],[4,5,6]])
    # print(a)
    # print(a.transpose())

    # # Dot product
    # print(u)
    # print(v.dot(u))

    # # Determinant
    c = Matrix([[42, 1, 0], [-5, 1, 0], [0, 0, 1]])
    # print(c.det())
    # print(Matrix([[3,1,4],[1,5,9],[2,6,5]]).det())

    #Elementary operations
    # print(-m)
    # print(m+n)
    # print(m-n)
    # print(5*m)
    # print(m*n)

def empty1d(length):
    return [None] * length

def empty2d(rows, cols):
    return [empty1d(cols) for i in range(rows)]

Scalar = numbers.Complex

class Matrix():
    # CREATION AND MUTATION
    def __init__(self, values):
        assert values != [[]]  # There shall be only one representation of a 0x0 matrix: values == []
        self.__rows = len(values)
        self.__cols = len(values[0]) if self.rows > 0 else 0
        for row in values:
            assert len(row) == self.cols  # Ensures that the array is not jagged
            for value in row:
                assert isinstance(value, Scalar)  # Ensures that all elements are numbers
        self.values = copy.deepcopy(values)

    @staticmethod
    def vector_from_list(values):
        return Matrix([values]).transpose()
    
    def vector_to_list(self):
        assert self.is_vector()
        return self.transpose().values[0]
    
    def is_vector(self):
        return self.cols == 1

    def __getitem__(self, key):
        i,j = key
        if isinstance(i, slice) or isinstance(j, slice):
            sliced_rows = self.values[i]
            if len(sliced_rows) == 0: return []
            if type(sliced_rows[0]) is not list: sliced_rows = [sliced_rows]
            result = Matrix([(row[j] if type(row[j]) is list else [row[j]]) for row in sliced_rows])
            if result.rows == 1 and result.cols == 1: return result[0][0]
            else: return result
        else:
            return self.values[i][j]
    
    def __setitem__(self, key, value):
        i,j = key
        if isinstance(value, Matrix): value = value.values
        if isinstance(i, slice) or isinstance(j, slice):
            sliced_rows = self.values[i]
            if len(sliced_rows) == 0: return []
            if type(sliced_rows[0]) is not list: sliced_rows = [sliced_rows]
            for x, row in enumerate(sliced_rows): row[j] = value[x][j]
        else:
            self.values[i][j] = value
    
    def copy(self):
        return copy.deepcopy(self)
    
    #PROPERTIES
    @property
    def rows(self):
        return self.__rows
    
    @property
    def cols(self):
        return self.__cols
    
    @property
    def length(self):
        return self.rows
    
    def __len__(self):
        return self.length

    def __repr__(self):
        return "Matrix("+str(self.values)+")"
    
    def __str__(self):
        return "["+" \n ".join(["["+", ".join([str(v) for v in row])+"]" for row in self.values])+"]"
    
    #ELEMENT-WISE OPERATIONS
    def __iter__(self):
        return self.MatrixIterator(self)
    
    class MatrixIterator():
        def __init__(self, matrix):
            self.matrix = matrix
            self.row = 0
            self.col = 0
        
        def __next__(self):
            try: result = self.matrix[self.row, self.col]
            except IndexError: raise StopIteration
            self.col += 1
            if self.col >= self.matrix.cols:
                self.col = 0
                self.row += 1
            return result
        

    def apply(self, f, inplace=False):
        if not inplace: 
            result = self.copy()
            result.apply(f, inplace=True)
            return result
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self[i,j] = f(self[i,j])
    
    def apply_binary(self, other, f, inplace=False):
        if not inplace: 
            result = self.copy()
            result.apply_binary(other, f, inplace=True)
            return result
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.values[i][j] = f(self.values[i][j], other.values[i][j])
    
    #ELEMENTARY UNARY OPERATIONS
    def __pos__(self): return self.apply(lambda x: +x)
    def __neg__(self): return self.apply(lambda x: -x)
    def __abs__(self): return self.apply(lambda x: abs(x))

    def transpose(self):
        result = empty2d(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j][i] = self.values[i][j]
        return Matrix(result)
    
    def det(self):
        assert self.rows == self.cols
        if self.rows == 0:
            return 1
        result = 0
        for i in range(self.rows):
            sub = Matrix([[e for c, e in enumerate(row) if c != 0] for r, row in enumerate(self.values) if r != i])
            d = sub.det()
            x = ((-1)**i)*self.values[i][0]*d
            result += x
        return result

    #ELEMENTARY BINARY OPERATIONS
    def __add__(self, other): return self.apply_binary(other, lambda x,y: x+y)
    def __sub__(self, other): return self.apply_binary(other, lambda x,y: x-y)

    def dot(self, other):
        assert self.is_vector()
        assert other.is_vector()
        return sum(self.apply_binary(other, lambda x,y: x*y))

    def __mul__(self, other):
        if isinstance(other, Scalar):
            return self.apply(lambda x: other*x)
        else:
            assert self.cols == other.rows
            result = empty2d(other.cols, self.rows)
            for i in range(self.rows):
                for j in range(other.cols):
                    result[i][j] = self[i,:].transpose().dot(other[:,j])
            return Matrix(result)

    def __rmul__(self, other):
        assert isinstance(other, Scalar)
        return self*other
    
    def __div__(self, other):
        assert isinstance(other, Scalar)
        return self.apply(lambda x: x/other)

if __name__ == "__main__":
    main()
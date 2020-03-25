import time
import numbers
import copy

def main():
    # Real vectors
    v = Vector([1, 2, 3])
    # print(v)

    # Complex vectors
    u = Vector([4+1j, 5-1j, 6])
    # print(u)

    # Real matrices
    n = Matrix([[1, 2],[3, 4]])
    print(n)

    # Complex matrices
    m = Matrix([[1+2j, 3+4j], [5+6j, 7+8j]])
    # print(m)

    # Transposition
    a = Matrix([[1,2,3],[4,5,6]])
    # print(a)
    # print(a.transpose())

    # Vector as matrix
    # print(v.to_matrix())

    # Dot product
    # print(u)
    # print(v.dot(u))

    # Matrix multiplication
    b = Matrix([[7, 8], [9, 0], [1,2]])
    # print(a*b)

    # Column vectors
    # print(a.col_vectors)

    # Determinant
    c = Matrix([[42, 1, 0], [-5, 1, 0], [0, 0, 1]])
    # print(c.det())
    # print(Matrix([[3,1,4],[1,5,9],[2,6,5]]).det())

def empty1d(length):
    return [None] * length

def empty2d(rows, cols):
    return [empty1d(cols) for i in range(rows)]

Scalar = numbers.Complex

class Vector():
    def __init__(self, values):
        self._length = len(values)
        for value in values:
            assert isinstance(value, Scalar)  # Ensures that all elements are numbers
        self.values = values
    
    @property
    def length(self):
        return self._length
    
    def __len__(self):
        return self.length

    def __repr__(self):
        return "Vector("+str(self.values)+")"

    def __str__(self):
        return "<"+", ".join([str(v) for v in self.values])+">"

    def to_matrix(self, type="column"):
        if type == "row": return Matrix([self.values])
        if type == "column": return self.to_matrix("row").transpose()

    def dot(self, other):
        if isinstance(other, Vector):
            assert len(self) == len(other)
            sum = 0
            for i in range(self.length):
                sum += self.values[i]*other.values[i]
            return sum
    
    def __mul__(self, other):
        if isinstance(other, Scalar):
            result = self.values.deepcopy()


class Matrix():
    def __init__(self, values):
        self._rows = len(values)
        self._cols = len(values[0]) if self.rows > 0 else 0
        for row in values:
            assert len(row) == self.cols  # Ensures that the array is not jagged
            for value in row:
                assert isinstance(value, Scalar)  # Ensures that all elements are numbers
        self.values = copy.deepcopy(values)
    
    @property
    def rows(self):
        return self._rows
    
    @property
    def cols(self):
        return self._cols

    def __repr__(self):
        return "Matrix("+str(self.values)+")"
    
    def __str__(self):
        return "["+" \n ".join(["["+", ".join([str(v) for v in row])+"]" for row in self.values])+"]"
    
    @property
    def row_vectors(self):
        return [Vector(row) for row in self.values]
    
    @property
    def col_vectors(self):
        return [Vector(col) for col in self.transpose().values]

    def transpose(self):
        result = empty2d(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j][i] = self.values[i][j]
        return Matrix(result)
    
    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.rows == other.rows and self.cols == other.cols
            result = empty2d(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i][j] += other.values[i][j]
            return Matrix(result)
    
    def __mul__(self, other):
        if isinstance(other, Scalar):
            result = empty2d(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i][j] *= other
            return Matrix(result)
        elif isinstance(other, Matrix):
            assert self.cols == other.rows
            result = empty2d(other.cols, self.rows)
            for i in range(self.rows):
                for j in range(other.cols):
                    result[i][j] = self.row_vectors[i].dot(other.col_vectors[j])
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

if __name__ == "__main__":
    main()
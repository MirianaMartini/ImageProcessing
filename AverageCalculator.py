
"""
Eseguire lo script per ogni path
"""

path = "Test/TestRoberta.txt"
#path = "TestSigned/TestRoberta.txt"
#path = "Test/TestMiriana.txt"
#path = "TestSigned/TestMiriana.txt"

tot = 100


def average():
    lines = []

    try:
        with open(path, "r+") as f:
            i = 0
            sum = 0

            lines = f.readlines()
            f.close()

            for line in lines:
                x = line.split(': ')
                x = x[1].split('/')
                sum += int(x[0])
                i += 1
            average = sum/i
            print("Average '{}': {}/{}\n".format(path, average, tot))

    except IOError:
        print("No File Existing")


if __name__ == "__main__":
    try:
        average()
    except KeyboardInterrupt:
        exit(0)

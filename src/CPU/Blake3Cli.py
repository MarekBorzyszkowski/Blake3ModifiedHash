import sys

from HashFunc import blake3_hash, message_to_binary


print("To end the program insert EOF (Ctrl + D on unix, Ctrl + Z on windows)")
while True:
    a = sys.stdin.readline()
    if len(a) == 0:
        break
    a = a.strip('\n')
    binary_message = message_to_binary(a)
    results = blake3_hash(binary_message)
    print(f"Blake3 hash of '{a}' = ", end="")
    for result in results:
        print(f'{result:04x}'.upper(), end=" ")
    print()
print("Thank you for using Blake3 hash!")
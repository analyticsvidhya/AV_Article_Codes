def reverse(x: int) -> int:
    """
    Given a 32-bit signed integer, reverse digits of an integer.
    """
    str_num = str(x)
    is_negative = False
    if str_num[0] == '-':
        is_negative = True
        str_num = str_num[1:]

    sign = '-' if is_negative else '+'

    num = int(sign + "".join(list(reversed(str_num))))

    return num
    
print(reverse(123))

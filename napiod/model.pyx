def state_variable_to_state_index(
        int price_change,
        int imbalance,
        int num_discretisations
        ):
    """
    `price_change` takes values -1, 0, +1, indicating price decrease, price unchanged, price increase.
    Assuming `num_discretisations` is odd, `imbalance` takes values from -(num_discretisations - 1) / 2 to +num_discretisations - 1)/2. 

    """
    return (price_change + 1) * num_discretisations + (num_discretisations // 2 + imbalance)

def price_change_from_state_index(
        int x,
        int num_discretisations
        ):
    cdef int price_change = -1 + (x // num_discretisations)
    return price_change

def imbalance_from_state_index(
        int x,
        int num_discretisations
        ):
    cdef int imbalance = (x %  num_discretisations) - (num_discretisations // 2)
    return imbalance

def inflationary_state_indexes(
        int num_discretisations
        ):
    for imbalance in range(-(num_discretisations //2), 1 + (num_discretisations // 2)):
        yield state_variable_to_state_index(1, imbalance, num_discretisations)

def _inflationary_state_indexes_recon(
        int num_discretisations
        ):
    for x in range(2*num_discretisations, 3*num_discretisations):
        yield x


def deflationary_state_indexes(
        int num_discretisations
        ):
    for imbalance in range(-(num_discretisations //2), 1+ (num_discretisations // 2)):
        yield state_variable_to_state_index(-1, imbalance, num_discretisations)


def _deflationary_state_indexes_recon(
        int num_discretisations
        ):
    for x in range(num_discretisations):
        yield x

        

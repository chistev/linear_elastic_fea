class Material:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

# Example usage:
materials = {
    1: Material(E=200e9, nu=0.3),  # steel
    2: Material(E=70e9, nu=0.33),  # aluminum
}

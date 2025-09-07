class Material:
    def __init__(self, E, nu, rho):
        self.E = E
        self.nu = nu
        self.rho = rho  # Density (kg/m^3)

# Example usage:
materials = {
    1: Material(E=200e9, nu=0.3, rho=7850),  # Steel
    2: Material(E=70e9, nu=0.33, rho=2700),  # Aluminum
}
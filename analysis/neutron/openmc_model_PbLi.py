import openmc
from libra_toolbox.neutronics import vault
from libra_toolbox.neutronics.neutron_source import A325_generator_diamond
import math
import numpy as np

############################################################################
# Functions


def calculate_breeder_depth(R, r, g, V):
    """Calculates the height (H) of a cylindrical volume (radius R & volume V) with an
    inserted inner cylinder (of radius r & gap from larger cylinder floor of g).

    Args:
        R (float): Major radius of breeder volume cylinder (cm)
        r (float): Radius of inner cylinder inserted into breeder volume (cm)
        g (float): Gap between floor of breeder volume cylinder and inserted inner cylinder (cm)
        V (float): Volume of breeder material(cm3)

    Returns:
        (float): Depth of breeder material volume.
    """
    v1 = math.pi * R**2 * g  # Volume of cylinder beneath heater
    v2 = V - v1  # Volume of annulus around heater

    h1 = g  # Height below heater
    h2 = v2 / (math.pi * (R**2 - r**2))  # Height of annulus around heater

    H = h1 + h2  # Total height for given volume
    return H


def baby_model():
    """Returns an openmc model of the BABY experiment.

    Returns:
        the openmc model
    """

    materials = [
        SS316L,
        lithium_lead,
        SS304,
        heater_mat,
        furnace,
        alumina,
        lead,
        air,
        epoxy,
        he,
    ]

    sphere, PbLi_cell, cells = baby_geometry(x_c, y_c, z_c)

    ############################################################################
    # Define Settings

    settings = openmc.Settings()

    src = A325_generator_diamond((x_c, y_c, z_c - 5.635), (1, 0, 0))
    settings.source = src
    settings.batches = 100
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.particles = int(1e4)
    settings.output = {"tallies": False}
    settings.photon_transport = False

    ############################################################################
    overall_exclusion_region = -sphere

    ############################################################################
    # Specify Tallies

    # Create a list of tallies
    tallies = openmc.Tallies()

    # Create tally for entire Li2O cell for global TBR results
    tbr_tally = openmc.Tally(name="TBR")
    tbr_tally.scores = ["(n,Xt)"]
    tbr_tally.filters = [openmc.CellFilter(PbLi_cell)]  # Add cell filter to tally

    # Create a second tally to add the mesh filter to for spatial TBR distribution results
    tbr_tally_mesh = openmc.Tally(name="TBR_mesh")
    tbr_tally_mesh.scores = ["(n,Xt)"]
    tbr_tally_mesh.filters = [openmc.CellFilter(PbLi_cell)]
    # Add cell filter to tally_mesh

    # Create a cylindrical mesh
    r_grid = np.linspace(
        0, PbLi_radius, (int(PbLi_radius * 10)) + 1
    )  # ~0.1cm radial bins (10x as many bins as breeder radius in cm)

    phi_grid = (0, 2 * np.pi)  # 1 azimuthal bin to capture full 360 degrees

    z_grid = np.linspace(
        0, PbLi_thickness, (int(PbLi_thickness * 10)) + 1
    )  # ~0.1cm axial bins (10x as many bins as breeder depth in cm)

    mesh_origin = (
        x_c,
        y_c,
        PbLi_z,
    )  # Origin of the mesh aligned with xy position of BABY central axis, and z position of the bottom of the Li2O bed.

    cyl_mesh = openmc.CylindricalMesh(r_grid, z_grid, phi_grid, mesh_origin)

    # Create a mesh filter from the cylindrical mesh
    mesh_filter = openmc.MeshFilter(cyl_mesh)

    # Add cylindrical mesh filter to tbr_tally_mesh
    tbr_tally_mesh.filters.append(mesh_filter)

    # Append both tallies to the list of tallies
    tallies.append(tbr_tally)
    tallies.append(tbr_tally_mesh)

    ############################################################################
    # Model

    model = vault.build_vault_model(
        settings=settings,
        tallies=tallies,
        added_cells=cells,
        added_materials=materials,
        overall_exclusion_region=overall_exclusion_region,
    )

    return model


def baby_geometry(x_c: float, y_c: float, z_c: float):
    """Returns the geometry for the BABY experiment.

    Args:
        x_c: x-coordinate of the center of the BABY experiment (cm)
        y_c: y-coordinate of the center of the BABY experiment (cm)
        z_c: z-coordinate of the center of the BABY experiment (cm)

    Returns:
        the sphere, cllif cell, and cells
    """

    ########## Surfaces ##########
    z_plane_1 = openmc.ZPlane(0.0 + z_c)
    z_plane_2 = openmc.ZPlane(epoxy_thickness + z_c)
    z_plane_3 = openmc.ZPlane(epoxy_thickness + alumina_compressed_thickness + z_c)
    z_plane_4 = openmc.ZPlane(
        epoxy_thickness + alumina_compressed_thickness + ov_base_thickness + z_c
    )
    z_plane_5 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + z_c
    )
    z_plane_6 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + z_c
    )
    z_plane_7 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + z_c
    )
    z_plane_8 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + PbLi_thickness
        + z_c
    )
    z_plane_9 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + PbLi_thickness
        + cover_he_thickness
        + z_c
    )
    z_plane_10 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + he_thickness
        + iv_base_thickness
        + PbLi_thickness
        + cover_he_thickness
        + iv_cap
        + z_c
    )
    z_plane_11 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + alumina_thickness
        + furnace_thickness
        + z_c
    )
    z_plane_12 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + ov_height
        + z_c
    )
    z_plane_13 = openmc.ZPlane(
        epoxy_thickness
        + alumina_compressed_thickness
        + ov_base_thickness
        + ov_height
        + ov_cap
        + z_c
    )
    z_plane_14 = openmc.ZPlane(z_c - table_height)
    z_plane_15 = openmc.ZPlane(z_c - table_height - epoxy_thickness)

    ########## Cylinder ##########
    z_cyl_1 = openmc.ZCylinder(x0=x_c, y0=y_c, r=PbLi_radius)
    z_cyl_2 = openmc.ZCylinder(x0=x_c, y0=y_c, r=iv_external_radius)
    z_cyl_3 = openmc.ZCylinder(x0=x_c, y0=y_c, r=he_radius)
    z_cyl_4 = openmc.ZCylinder(x0=x_c, y0=y_c, r=furnace_radius)
    z_cyl_5 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_internal_radius)
    z_cyl_6 = openmc.ZCylinder(x0=x_c, y0=y_c, r=ov_external_radius)

    right_cyl = openmc.model.RightCircularCylinder(
        (x_c, y_c, heater_z), heater_length, heater_radius, axis="z"
    )
    ext_cyl_source = openmc.model.RightCircularCylinder(
        (source_x, y_c, source_z), source_h, source_external_r, axis="x"
    )
    source_region = openmc.model.RightCircularCylinder(
        (source_x + 0.25, y_c, source_z), source_h - 0.50, source_internal_r, axis="x"
    )

    ########## Sphere ##########
    sphere = openmc.Sphere(x0=x_c, y0=y_c, z0=z_c, r=50.00)  # before r=50.00

    ########## Lead bricks positioned under the source ##########
    positions = [
        (x_c - 13.50, y_c, z_c - table_height),
        (x_c - 4.50, y_c, z_c - table_height),
        (x_c + 36.50, y_c, z_c - table_height),
        (x_c + 27.50, y_c, z_c - table_height),
    ]

    lead_blocks = []
    for position in positions:
        lead_block_region = openmc.model.RectangularParallelepiped(
            position[0] - lead_width / 2,
            position[0] + lead_width / 2,
            position[1] - lead_length / 2,
            position[1] + lead_length / 2,
            position[2],
            position[2] + lead_height,
        )
        lead_blocks.append(lead_block_region)

    ########## Regions ##########
    source_wall_region = -ext_cyl_source & +source_region
    source_region = -source_region
    epoxy_region = +z_plane_1 & -z_plane_2 & -sphere
    alumina_compressed_region = +z_plane_2 & -z_plane_3 & -sphere
    bottom_vessel = +z_plane_3 & -z_plane_4 & -z_cyl_6
    top_vessel = +z_plane_12 & -z_plane_13 & -z_cyl_6 & +right_cyl
    cylinder_vessel = +z_plane_4 & -z_plane_12 & +z_cyl_5 & -z_cyl_6
    vessel_region = bottom_vessel | cylinder_vessel | top_vessel
    alumina_region = +z_plane_4 & -z_plane_5 & -z_cyl_5
    bottom_cap = +z_plane_6 & -z_plane_7 & -z_cyl_2 & +right_cyl
    cylinder_cap = +z_plane_7 & -z_plane_9 & +z_cyl_1 & -z_cyl_2 & +right_cyl
    top_cap = +z_plane_9 & -z_plane_10 & -z_cyl_2 & +right_cyl
    cap_region = bottom_cap | cylinder_cap | top_cap
    PbLi_region = +z_plane_7 & -z_plane_8 & -z_cyl_1 & +right_cyl
    gap_region = +z_plane_8 & -z_plane_9 & -z_cyl_1 & +right_cyl
    furnace_region = +z_plane_5 & -z_plane_11 & +z_cyl_3 & -z_cyl_4
    heater_region = -right_cyl
    table_under_source_region = +z_plane_15 & -z_plane_14 & -sphere
    lead_block_1_region = -lead_blocks[0]
    lead_block_2_region = -lead_blocks[1]
    lead_block_3_region = -lead_blocks[2]
    lead_block_4_region = -lead_blocks[3]
    he_region = (
        +z_plane_5
        & -z_plane_12
        & -z_cyl_5
        & ~source_region
        & ~epoxy_region
        & ~alumina_compressed_region
        & ~alumina_region
        & ~PbLi_region
        & ~gap_region
        & ~furnace_region
        & ~vessel_region
        & ~cap_region
        & ~heater_region
        & ~table_under_source_region
        & ~lead_block_1_region
        & ~lead_block_2_region
        & ~lead_block_3_region
        & ~lead_block_4_region
    )
    sphere_region = (
        -sphere
        & ~source_wall_region
        & ~source_region
        & ~epoxy_region
        & ~alumina_compressed_region
        & ~alumina_region
        & ~PbLi_region
        & ~gap_region
        & ~furnace_region
        & ~he_region
        & ~vessel_region
        & ~cap_region
        & ~heater_region
        & ~table_under_source_region
        & ~lead_block_1_region
        & ~lead_block_2_region
        & ~lead_block_3_region
        & ~lead_block_4_region
    )

    # cells
    source_wall_cell_1 = openmc.Cell(region=source_wall_region)
    source_wall_cell_1.fill = SS304
    source_region = openmc.Cell(region=source_region)
    source_region.fill = None
    epoxy_cell = openmc.Cell(region=epoxy_region)
    epoxy_cell.fill = epoxy
    alumina_compressed_cell = openmc.Cell(region=alumina_compressed_region)
    alumina_compressed_cell.fill = alumina
    vessel_cell = openmc.Cell(region=vessel_region)
    vessel_cell.fill = SS316L
    alumina_cell = openmc.Cell(region=alumina_region)
    alumina_cell.fill = alumina
    PbLi_cell = openmc.Cell(region=PbLi_region)
    PbLi_cell.fill = lithium_lead  # cllif_nat or lithium_lead
    gap_cell = openmc.Cell(region=gap_region)
    gap_cell.fill = he
    cap_cell = openmc.Cell(region=cap_region)
    cap_cell.fill = SS316L
    furnace_cell = openmc.Cell(region=furnace_region)
    furnace_cell.fill = furnace
    heater_cell = openmc.Cell(region=heater_region)
    heater_cell.fill = heater_mat
    table_cell = openmc.Cell(region=table_under_source_region)
    table_cell.fill = epoxy
    sphere_cell = openmc.Cell(region=sphere_region)
    sphere_cell.fill = air
    he_cell = openmc.Cell(region=he_region)
    he_cell.fill = he
    lead_block_1_cell = openmc.Cell(region=lead_block_1_region)
    lead_block_1_cell.fill = lead
    lead_block_2_cell = openmc.Cell(region=lead_block_2_region)
    lead_block_2_cell.fill = lead
    lead_block_3_cell = openmc.Cell(region=lead_block_3_region)
    lead_block_3_cell.fill = lead
    lead_block_4_cell = openmc.Cell(region=lead_block_4_region)
    lead_block_4_cell.fill = lead

    cells = [
        source_wall_cell_1,
        source_region,
        epoxy_cell,
        alumina_compressed_cell,
        vessel_cell,
        alumina_cell,
        cap_cell,
        PbLi_cell,
        gap_cell,
        furnace_cell,
        heater_cell,
        he_cell,
        sphere_cell,
        table_cell,
        lead_block_1_cell,
        lead_block_2_cell,
        lead_block_3_cell,
        lead_block_4_cell,
    ]

    return sphere, PbLi_cell, cells


############################################################################
# Dimensions
# All dimensions in cm

# BABY coordinates within vault
x_c = 587  # cm
y_c = 60  # cm
z_c = 100  # cm

## BABY vertical dimensions
epoxy_thickness = 2.54  # 1 inch
alumina_compressed_thickness = 2.54  # 1 inch
ov_base_thickness = 0.786
alumina_thickness = 0.635
he_thickness = 0.6
iv_base_thickness = 0.3
heater_gap = 0.878
iv_height = 10.8903
iv_cap = 1.422
furnace_thickness = 15.24
ov_height = 21.093
ov_cap = 2.392
table_height = 28.00
lead_height = 4.00
lead_width = 8.00
lead_length = 16.00

heater_length = 25.40

PbLi_z = (
    epoxy_thickness
    + alumina_compressed_thickness
    + ov_base_thickness
    + alumina_thickness
    + he_thickness
    + iv_base_thickness
    + z_c
)

heater_z = PbLi_z + heater_gap

## BABY Radial dimensions
heater_radius = 0.439
PbLi_radius = 7.00
iv_external_radius = 7.3
he_radius = 9.144
furnace_radius = 14.224
ov_internal_radius = 17.561
ov_external_radius = 17.78

## Calculated dimensions
PbLi_volume = 1000  # 1L = 1000 cm3

PbLi_thickness = calculate_breeder_depth(
    PbLi_radius, heater_radius, heater_gap, PbLi_volume
)
cover_he_thickness = (
    iv_height - PbLi_thickness
)  # Gap between surface of PbLi and top boundary of inner vessel.

## Source dimensions
source_h = 50.00
source_x = x_c - 13.50
source_z = z_c - 5.635
source_external_r = 5.00
source_internal_r = 4.75


############################################################################
# Define Materials
# Source: PNNL Materials Compendium April 2021
# PNNL-15870, Rev. 2

# 316L Stainless Steel
# Data from https://www.thyssenkrupp-materials.co.uk/stainless-steel-316l-14404.html
SS316L = openmc.Material(name="316L Steel")
SS316L.add_element("C", 0.0003, "wo")
SS316L.add_element("Si", 0.01, "wo")
SS316L.add_element("Mn", 0.02, "wo")
SS316L.add_element("P", 0.00045, "wo")
SS316L.add_element("S", 0.000151, "wo")
SS316L.add_element("Cr", 0.175, "wo")
SS316L.add_element("Ni", 0.115, "wo")
SS316L.add_element("N", 0.001, "wo")
SS316L.add_element("Mo", 0.00225, "wo")
SS316L.add_element("Fe", 0.655599, "wo")

# Lithium-Lead
# Composition from certificate of analysis provided with Lithium-Lead from Camex
lithium_lead = openmc.Material(name="Lithium Lead")
lithium_lead.add_element("Pb", 0.993479, "wo")
lithium_lead.add_element("Li", 0.0064, "wo")
lithium_lead.add_element("Tl", 0.00002, "wo")
lithium_lead.add_element("Zn", 0.000002, "wo")
lithium_lead.add_element("Sn", 0.000002, "wo")
lithium_lead.add_element("Sb", 0.000002, "wo")
lithium_lead.add_element("Ni", 0.000001, "wo")
lithium_lead.add_element("Cu", 0.000002, "wo")
lithium_lead.add_element("Cd", 0.000002, "wo")
lithium_lead.add_element("Bi", 0.00008, "wo")
lithium_lead.add_element("As", 0.000002, "wo")
lithium_lead.add_element("Ag", 0.000008, "wo")
lithium_lead.set_density("g/cm3", 9.10411395)  # Density at 600C

# Stainless Steel 304 from PNNL Materials Compendium (PNNL-15870 Rev2)
SS304 = openmc.Material(name="Stainless Steel 304")
# SS304.temperature = 700 + 273
SS304.add_element("C", 0.000800, "wo")
SS304.add_element("Mn", 0.020000, "wo")
SS304.add_element("P", 0.000450, "wo")
SS304.add_element("S", 0.000300, "wo")
SS304.add_element("Si", 0.010000, "wo")
SS304.add_element("Cr", 0.190000, "wo")
SS304.add_element("Ni", 0.095000, "wo")
SS304.add_element("Fe", 0.683450, "wo")
SS304.set_density("g/cm3", 8.00)

# Central heater material
# Data from where???
heater_mat = openmc.Material(name="heater")
heater_mat.add_element("C", 0.000990, "wo")
heater_mat.add_element("Al", 0.003960, "wo")
heater_mat.add_element("Si", 0.004950, "wo")
heater_mat.add_element("P", 0.000148, "wo")
heater_mat.add_element("S", 0.000148, "wo")
heater_mat.add_element("Ti", 0.003960, "wo")
heater_mat.add_element("Cr", 0.215000, "wo")
heater_mat.add_element("Mn", 0.004950, "wo")
heater_mat.add_element("Fe", 0.049495, "wo")
heater_mat.add_element("Co", 0.009899, "wo")
heater_mat.add_element("Ni", 0.580000, "wo")
heater_mat.add_element("Nb", 0.036500, "wo")
heater_mat.add_element("Mo", 0.090000, "wo")
heater_mat.set_density("g/cm3", 2.44)

# Using Microtherm with 1 a% Al2O3, 27 a% ZrO2, and 72 a% SiO2
# https://www.foundryservice.com/product/microporous-silica-insulating-boards-mintherm-microtherm-1925of-grades/
furnace = openmc.Material(name="Furnace")
# Estimate average temperature of furnace insulation to be around 300 C
# furnace.temperature = 273 + 300
furnace.add_element("Al", 0.004, "ao")
furnace.add_element("O", 0.666, "ao")
furnace.add_element("Si", 0.240, "ao")
furnace.add_element("Zr", 0.090, "ao")
furnace.set_density("g/cm3", 0.30)

# alumina insulation
# data from https://precision-ceramics.com/materials/alumina/
alumina = openmc.Material(name="Alumina insulation")
alumina.add_element("O", 0.6, "ao")
alumina.add_element("Al", 0.4, "ao")
alumina.set_density("g/cm3", 3.98)

# air
air = openmc.Material(name="Air")
air.add_element("C", 0.00012399, "wo")
air.add_element("N", 0.75527, "wo")
air.add_element("O", 0.23178, "wo")
air.add_element("Ar", 0.012827, "wo")
air.set_density("g/cm3", 0.0012)

# epoxy
# Data from where??
epoxy = openmc.Material(name="Epoxy")
epoxy.add_element("C", 0.70, "wo")
epoxy.add_element("H", 0.08, "wo")
epoxy.add_element("O", 0.15, "wo")
epoxy.add_element("N", 0.07, "wo")
epoxy.set_density("g/cm3", 1.2)

# helium @5psig
pressure = 34473.8  # Pa ~ 5 psig
temperature = 300  # K
R_he = 2077  # J/(kg*K)
density = pressure / (R_he * temperature) / 1000  # in g/cm^3
he = openmc.Material(name="Helium")
he.add_element("He", 1.0, "ao")
he.set_density("g/cm3", density)

# lead
# data from https://wwwrcamnl.wr.usgs.gov/isoig/period/pb_iig.html
lead = openmc.Material()
lead.set_density("g/cm3", 11.34)
lead.add_nuclide("Pb204", 0.014, "ao")
lead.add_nuclide("Pb206", 0.241, "ao")
lead.add_nuclide("Pb207", 0.221, "ao")
lead.add_nuclide("Pb208", 0.524, "ao")

############################################################################
# Main

if __name__ == "__main__":
    model = baby_model()
    model.run()
    sp = openmc.StatePoint(f"statepoint.{model.settings.batches}.h5")
    tbr_tally = sp.get_tally(name="TBR").get_pandas_dataframe()

    print(f"TBR: {tbr_tally['mean'].iloc[0] :.6e}\n")
    print(f"TBR std. dev.: {tbr_tally['std. dev.'].iloc[0] :.6e}\n")

    processed_data = {
        "modelled_TBR": {
            "mean": tbr_tally["mean"].iloc[0],
            "std_dev": tbr_tally["std. dev."].iloc[0],
        }
    }

    import json

    processed_data_file = "../../data/processed_data.json"

    try:
        with open(processed_data_file, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        print(f"Processed data file not found, creating it in {processed_data_file}")
        existing_data = {}

    existing_data.update(processed_data)

    with open(processed_data_file, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Processed data stored in {processed_data_file}")

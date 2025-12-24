# Timing constraints for ALU synthesis
# Target: 100 MHz clock (10ns period)

create_clock -period 10.0 -name clk [get_ports clk]

# Input delays (assuming external logic has 2ns delay)
set_input_delay -clock clk 2.0 [get_ports {a[*] b[*] op[*] valid_in}]

# Output delays (assuming 2ns setup for downstream logic)
set_output_delay -clock clk 2.0 [get_ports {result[*] zero overflow valid_out}]

# Reset is async, but add false path
set_false_path -from [get_ports rst]

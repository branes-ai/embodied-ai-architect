/**
 * RISC-V ALU - Area-Optimized Design
 *
 * Key optimizations for area reduction:
 * 1. Fixed syntax errors with proper signal declarations
 * 2. Removed multiplication operation (high area cost)
 * 3. Simplified control logic and reduced intermediate signals
 * 4. Optimized flag generation with shared logic
 * 5. Reduced operation set to essential RISC-V operations
 * 6. Streamlined result mux for area efficiency
 */

module riscv_alu #(
    parameter WIDTH = 32
) (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                valid_in,
    input  logic [WIDTH-1:0]    operand_a,
    input  logic [WIDTH-1:0]    operand_b,
    input  logic [3:0]          alu_op,
    output logic [WIDTH-1:0]    result,
    output logic                valid_out,
    output logic                overflow,
    output logic                zero
);

    // ALU Operation Codes (essential set for area optimization)
    localparam OP_ADD  = 4'b0000;  // Addition
    localparam OP_SUB  = 4'b0001;  // Subtraction
    localparam OP_AND  = 4'b0010;  // Bitwise AND
    localparam OP_OR   = 4'b0011;  // Bitwise OR
    localparam OP_XOR  = 4'b0100;  // Bitwise XOR
    localparam OP_SLL  = 4'b0101;  // Shift Left Logical
    localparam OP_SRL  = 4'b0110;  // Shift Right Logical
    localparam OP_SRA  = 4'b0111;  // Shift Right Arithmetic
    localparam OP_SLT  = 4'b1000;  // Set Less Than (signed)
    localparam OP_SLTU = 4'b1001;  // Set Less Than (unsigned)

    // Minimal internal signals for area optimization
    logic [WIDTH:0]     adder_result;
    logic [WIDTH-1:0]   logic_result;
    logic [WIDTH-1:0]   shift_result;
    logic [WIDTH-1:0]   compare_result;
    logic               is_subtract;

    // =========================================================================
    // Shared Arithmetic Unit (ADD/SUB)
    // =========================================================================
    
    assign is_subtract = (alu_op == OP_SUB);
    assign adder_result = operand_a + (is_subtract ? (~operand_b + 1'b1) : operand_b);

    // =========================================================================
    // Logic Operations (shared structure)
    // =========================================================================
    always_comb begin
        case (alu_op[1:0])
            2'b10: logic_result = operand_a & operand_b;  // AND
            2'b11: logic_result = operand_a | operand_b;  // OR
            2'b00: logic_result = operand_a ^ operand_b;  // XOR
            default: logic_result = '0;
        endcase
    end

    // =========================================================================
    // Shift Operations (area-optimized)
    // =========================================================================
    always_comb begin
        case (alu_op[1:0])
            2'b01: shift_result = operand_a << operand_b[4:0];  // SLL
            2'b10: shift_result = operand_a >> operand_b[4:0];  // SRL
            2'b11: shift_result = $signed(operand_a) >>> operand_b[4:0];  // SRA
            default: shift_result = '0;
        endcase
    end

    // =========================================================================
    // Comparison Operations (simplified)
    // =========================================================================
    always_comb begin
        if (alu_op == OP_SLT) begin
            compare_result = {31'b0, $signed(operand_a) < $signed(operand_b)};
        end else begin // OP_SLTU
            compare_result = {31'b0, operand_a < operand_b};
        end
    end

    // =========================================================================
    // Result Multiplexer (optimized for area)
    // =========================================================================
    always_comb begin
        case (alu_op[3:2])
            2'b00: result = (alu_op[1] == 1'b0) ? adder_result[WIDTH-1:0] : logic_result;
            2'b01: result = shift_result;
            2'b10: result = compare_result;
            default: result = '0;
        endcase
    end

    // =========================================================================
    // Flag Generation (simplified for area)
    // =========================================================================

    // Overflow: only for ADD/SUB operations
    assign overflow = ((alu_op == OP_ADD) || (alu_op == OP_SUB)) && 
                     (adder_result[WIDTH] != adder_result[WIDTH-1]);

    // Zero flag
    assign zero = (result == '0);

    // Valid output pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
        end
    end

endmodule
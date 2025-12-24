/**
 * RISC-V ALU - A simple ALU for testing optimization loops.
 *
 * This design has intentional optimization opportunities:
 * 1. Deep combinational logic in multiplier path
 * 2. Unshared add/subtract operations
 * 3. Non-pipelined shifter
 *
 * Based on RISC-V RV32I instruction set ALU operations.
 *
 * Integration Note:
 * This design is compatible with systars PE components.
 * The MAC operation (a * b + c) can be mapped to systars PE dataflow.
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

    // ALU Operation Codes (RISC-V inspired)
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
    localparam OP_MUL  = 4'b1010;  // Multiplication (lower 32 bits)
    localparam OP_MAC  = 4'b1011;  // Multiply-Accumulate (for systars compatibility)

    // Internal signals
    logic [WIDTH-1:0]   add_result;
    logic [WIDTH-1:0]   sub_result;
    logic [WIDTH-1:0]   and_result;
    logic [WIDTH-1:0]   or_result;
    logic [WIDTH-1:0]   xor_result;
    logic [WIDTH-1:0]   sll_result;
    logic [WIDTH-1:0]   srl_result;
    logic [WIDTH-1:0]   sra_result;
    logic [WIDTH-1:0]   slt_result;
    logic [WIDTH-1:0]   sltu_result;
    logic [2*WIDTH-1:0] mul_result;
    logic [WIDTH-1:0]   mac_result;

    // Overflow detection signals
    logic add_overflow;
    logic sub_overflow;

    // Accumulator for MAC operations (systars-style)
    logic [WIDTH-1:0] accumulator;

    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    // Addition with overflow detection
    assign {add_overflow, add_result} = {1'b0, operand_a} + {1'b0, operand_b};

    // Subtraction with overflow detection
    assign {sub_overflow, sub_result} = {1'b0, operand_a} - {1'b0, operand_b};

    // =========================================================================
    // Logical Operations
    // =========================================================================

    assign and_result = operand_a & operand_b;
    assign or_result  = operand_a | operand_b;
    assign xor_result = operand_a ^ operand_b;

    // =========================================================================
    // Shift Operations
    // Note: These are combinational barrel shifters - optimization opportunity!
    // =========================================================================

    assign sll_result = operand_a << operand_b[4:0];
    assign srl_result = operand_a >> operand_b[4:0];
    assign sra_result = $signed(operand_a) >>> operand_b[4:0];

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    assign slt_result  = {{(WIDTH-1){1'b0}}, $signed(operand_a) < $signed(operand_b)};
    assign sltu_result = {{(WIDTH-1){1'b0}}, operand_a < operand_b};

    // =========================================================================
    // Multiplication (Optimization target - deep combinational logic)
    // =========================================================================

    // Full multiplication - this creates a deep combinational path
    // Optimization: pipeline into stages or use systars PE
    assign mul_result = operand_a * operand_b;

    // =========================================================================
    // Multiply-Accumulate (systars PE compatible)
    // D = A * B + C (where C comes from accumulator)
    // =========================================================================

    assign mac_result = mul_result[WIDTH-1:0] + accumulator;

    // Accumulator register (for MAC chains)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
        end else if (valid_in && alu_op == OP_MAC) begin
            accumulator <= mac_result;
        end else if (valid_in && alu_op == OP_ADD) begin
            // Reset accumulator on regular ADD
            accumulator <= '0;
        end
    end

    // =========================================================================
    // Result Multiplexer
    // =========================================================================

    always_comb begin
        case (alu_op)
            OP_ADD:  result = add_result;
            OP_SUB:  result = sub_result;
            OP_AND:  result = and_result;
            OP_OR:   result = or_result;
            OP_XOR:  result = xor_result;
            OP_SLL:  result = sll_result;
            OP_SRL:  result = srl_result;
            OP_SRA:  result = sra_result;
            OP_SLT:  result = slt_result;
            OP_SLTU: result = sltu_result;
            OP_MUL:  result = mul_result[WIDTH-1:0];
            OP_MAC:  result = mac_result;
            default: result = '0;
        endcase
    end

    // =========================================================================
    // Output Flags
    // =========================================================================

    // Overflow: only meaningful for add/sub
    always_comb begin
        case (alu_op)
            OP_ADD:  overflow = add_overflow;
            OP_SUB:  overflow = sub_overflow;
            default: overflow = 1'b0;
        endcase
    end

    // Zero flag
    assign zero = (result == '0);

    // Valid output (registered for timing)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
        end
    end

endmodule

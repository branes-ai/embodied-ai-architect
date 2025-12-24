// Baseline ALU - deliberately suboptimal for optimization demonstration
// This design has several optimization opportunities:
// 1. Redundant logic in the operation decoder
// 2. No resource sharing between add/sub
// 3. Inefficient comparison implementation
// 4. Synchronous clear could be async for area savings

module alu_baseline #(
    parameter WIDTH = 8
)(
    input  wire                clk,
    input  wire                rst,
    input  wire [WIDTH-1:0]    a,
    input  wire [WIDTH-1:0]    b,
    input  wire [3:0]          op,      // Operation select
    input  wire                valid_in,
    output reg  [WIDTH-1:0]    result,
    output reg                 zero,
    output reg                 overflow,
    output reg                 valid_out
);

    // Operation codes
    localparam OP_ADD  = 4'b0000;
    localparam OP_SUB  = 4'b0001;
    localparam OP_AND  = 4'b0010;
    localparam OP_OR   = 4'b0011;
    localparam OP_XOR  = 4'b0100;
    localparam OP_SHL  = 4'b0101;  // Shift left
    localparam OP_SHR  = 4'b0110;  // Shift right
    localparam OP_LT   = 4'b0111;  // Less than (signed)
    localparam OP_EQ   = 4'b1000;  // Equal
    localparam OP_PASS = 4'b1001;  // Pass A through

    // Internal signals - SUBOPTIMAL: separate adder and subtractor
    wire [WIDTH:0] add_result;
    wire [WIDTH:0] sub_result;
    
    // SUBOPTIMAL: Two separate adders instead of one with carry-in
    assign add_result = {1'b0, a} + {1'b0, b};
    assign sub_result = {1'b0, a} - {1'b0, b};
    
    // SUBOPTIMAL: Comparison uses separate subtraction
    wire signed [WIDTH-1:0] a_signed = a;
    wire signed [WIDTH-1:0] b_signed = b;
    wire lt_result = (a_signed < b_signed);
    
    // SUBOPTIMAL: Equality check doesn't reuse subtraction result
    wire eq_result = (a == b);
    
    // Result computation
    reg [WIDTH-1:0] result_comb;
    reg overflow_comb;
    
    always @(*) begin
        result_comb = {WIDTH{1'b0}};
        overflow_comb = 1'b0;
        
        case (op)
            OP_ADD: begin
                result_comb = add_result[WIDTH-1:0];
                overflow_comb = add_result[WIDTH];
            end
            
            OP_SUB: begin
                result_comb = sub_result[WIDTH-1:0];
                overflow_comb = sub_result[WIDTH];
            end
            
            OP_AND: begin
                result_comb = a & b;
            end
            
            OP_OR: begin
                result_comb = a | b;
            end
            
            OP_XOR: begin
                result_comb = a ^ b;
            end
            
            OP_SHL: begin
                // SUBOPTIMAL: Full barrel shifter when we only need shift by 1
                result_comb = a << b[2:0];
            end
            
            OP_SHR: begin
                result_comb = a >> b[2:0];
            end
            
            OP_LT: begin
                result_comb = {{(WIDTH-1){1'b0}}, lt_result};
            end
            
            OP_EQ: begin
                result_comb = {{(WIDTH-1){1'b0}}, eq_result};
            end
            
            OP_PASS: begin
                result_comb = a;
            end
            
            default: begin
                result_comb = {WIDTH{1'b0}};
            end
        endcase
    end
    
    // SUBOPTIMAL: Synchronous reset uses more resources
    always @(posedge clk) begin
        if (rst) begin
            result    <= {WIDTH{1'b0}};
            zero      <= 1'b0;
            overflow  <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            result    <= result_comb;
            zero      <= (result_comb == {WIDTH{1'b0}});
            overflow  <= overflow_comb;
            valid_out <= valid_in;
        end
    end

endmodule

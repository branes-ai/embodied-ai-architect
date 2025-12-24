/**
 * Testbench for RISC-V ALU
 *
 * Tests all ALU operations and reports pass/fail counts
 * in a format the validation node can parse.
 */

`timescale 1ns/1ps

module riscv_alu_tb;

    // Parameters
    localparam WIDTH = 32;
    localparam CLK_PERIOD = 10;

    // DUT signals
    logic               clk;
    logic               rst_n;
    logic               valid_in;
    logic [WIDTH-1:0]   operand_a;
    logic [WIDTH-1:0]   operand_b;
    logic [3:0]         alu_op;
    logic [WIDTH-1:0]   result;
    logic               valid_out;
    logic               overflow;
    logic               zero;

    // Test counters
    integer tests_passed = 0;
    integer tests_failed = 0;

    // ALU opcodes
    localparam OP_ADD  = 4'b0000;
    localparam OP_SUB  = 4'b0001;
    localparam OP_AND  = 4'b0010;
    localparam OP_OR   = 4'b0011;
    localparam OP_XOR  = 4'b0100;
    localparam OP_SLL  = 4'b0101;
    localparam OP_SRL  = 4'b0110;
    localparam OP_SRA  = 4'b0111;
    localparam OP_SLT  = 4'b1000;
    localparam OP_SLTU = 4'b1001;
    localparam OP_MUL  = 4'b1010;
    localparam OP_MAC  = 4'b1011;

    // DUT instantiation
    riscv_alu #(
        .WIDTH(WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .alu_op(alu_op),
        .result(result),
        .valid_out(valid_out),
        .overflow(overflow),
        .zero(zero)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test task
    task automatic test_operation(
        input string name,
        input [3:0] op,
        input [WIDTH-1:0] a,
        input [WIDTH-1:0] b,
        input [WIDTH-1:0] expected
    );
        @(posedge clk);
        valid_in = 1;
        alu_op = op;
        operand_a = a;
        operand_b = b;

        @(posedge clk);
        valid_in = 0;

        @(posedge clk);  // Wait for result

        if (result === expected) begin
            tests_passed++;
            $display("PASS: %s - a=%h, b=%h, result=%h", name, a, b, result);
        end else begin
            tests_failed++;
            $display("FAIL: %s - a=%h, b=%h, expected=%h, got=%h",
                     name, a, b, expected, result);
        end
    endtask

    // Main test sequence
    initial begin
        // Initialize
        rst_n = 0;
        valid_in = 0;
        operand_a = 0;
        operand_b = 0;
        alu_op = 0;

        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("\n========================================");
        $display("RISC-V ALU Test Suite");
        $display("========================================\n");

        // Addition tests
        $display("--- Addition Tests ---");
        test_operation("ADD basic",     OP_ADD, 32'h00000005, 32'h00000003, 32'h00000008);
        test_operation("ADD zero",      OP_ADD, 32'h00000000, 32'h00000000, 32'h00000000);
        test_operation("ADD negative",  OP_ADD, 32'hFFFFFFFF, 32'h00000001, 32'h00000000);
        test_operation("ADD large",     OP_ADD, 32'h7FFFFFFF, 32'h00000001, 32'h80000000);

        // Subtraction tests
        $display("\n--- Subtraction Tests ---");
        test_operation("SUB basic",     OP_SUB, 32'h00000008, 32'h00000003, 32'h00000005);
        test_operation("SUB zero",      OP_SUB, 32'h00000005, 32'h00000005, 32'h00000000);
        test_operation("SUB negative",  OP_SUB, 32'h00000000, 32'h00000001, 32'hFFFFFFFF);

        // Logical tests
        $display("\n--- Logical Tests ---");
        test_operation("AND",           OP_AND, 32'hFF00FF00, 32'h0F0F0F0F, 32'h0F000F00);
        test_operation("OR",            OP_OR,  32'hFF00FF00, 32'h0F0F0F0F, 32'hFF0FFF0F);
        test_operation("XOR",           OP_XOR, 32'hFFFFFFFF, 32'hAAAAAAAA, 32'h55555555);

        // Shift tests
        $display("\n--- Shift Tests ---");
        test_operation("SLL by 1",      OP_SLL, 32'h00000001, 32'h00000001, 32'h00000002);
        test_operation("SLL by 4",      OP_SLL, 32'h0000000F, 32'h00000004, 32'h000000F0);
        test_operation("SRL by 1",      OP_SRL, 32'h80000000, 32'h00000001, 32'h40000000);
        test_operation("SRA by 4",      OP_SRA, 32'hF0000000, 32'h00000004, 32'hFF000000);

        // Comparison tests
        $display("\n--- Comparison Tests ---");
        test_operation("SLT true",      OP_SLT,  32'hFFFFFFFF, 32'h00000001, 32'h00000001);
        test_operation("SLT false",     OP_SLT,  32'h00000005, 32'h00000003, 32'h00000000);
        test_operation("SLTU true",     OP_SLTU, 32'h00000001, 32'hFFFFFFFF, 32'h00000001);
        test_operation("SLTU false",    OP_SLTU, 32'hFFFFFFFF, 32'h00000001, 32'h00000000);

        // Multiplication tests
        $display("\n--- Multiplication Tests ---");
        test_operation("MUL basic",     OP_MUL, 32'h00000003, 32'h00000004, 32'h0000000C);
        test_operation("MUL by zero",   OP_MUL, 32'h12345678, 32'h00000000, 32'h00000000);
        test_operation("MUL by one",    OP_MUL, 32'hABCDEF01, 32'h00000001, 32'hABCDEF01);

        // Summary
        repeat(5) @(posedge clk);

        $display("\n========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Passed: %0d", tests_passed);
        $display("Failed: %0d", tests_failed);
        $display("Total:  %0d", tests_passed + tests_failed);

        if (tests_failed == 0) begin
            $display("\nALL TESTS PASSED");
        end else begin
            $display("\nSOME TESTS FAILED");
        end

        $display("========================================\n");

        $finish;
    end

    // Timeout
    initial begin
        #100000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule

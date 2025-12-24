// Testbench for ALU verification
// Tests all operations with corner cases

`timescale 1ns/1ps

module alu_tb;
    parameter WIDTH = 8;
    
    reg                 clk;
    reg                 rst;
    reg  [WIDTH-1:0]    a;
    reg  [WIDTH-1:0]    b;
    reg  [3:0]          op;
    reg                 valid_in;
    wire [WIDTH-1:0]    result;
    wire                zero;
    wire                overflow;
    wire                valid_out;
    
    // Operation codes - must match DUT
    localparam OP_ADD  = 4'b0000;
    localparam OP_SUB  = 4'b0001;
    localparam OP_AND  = 4'b0010;
    localparam OP_OR   = 4'b0011;
    localparam OP_XOR  = 4'b0100;
    localparam OP_SHL  = 4'b0101;
    localparam OP_SHR  = 4'b0110;
    localparam OP_LT   = 4'b0111;
    localparam OP_EQ   = 4'b1000;
    localparam OP_PASS = 4'b1001;
    
    // Instantiate DUT - use generic name so same TB works for baseline and optimized
    alu_baseline #(.WIDTH(WIDTH)) dut (
        .clk(clk),
        .rst(rst),
        .a(a),
        .b(b),
        .op(op),
        .valid_in(valid_in),
        .result(result),
        .zero(zero),
        .overflow(overflow),
        .valid_out(valid_out)
    );
    
    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;
    
    // Test counters
    integer tests_passed = 0;
    integer tests_failed = 0;
    
    // Expected values
    reg [WIDTH-1:0] expected_result;
    reg expected_zero;
    reg expected_overflow;
    
    // Check task
    task check_result;
        input [WIDTH-1:0] exp_res;
        input exp_zero;
        input exp_ovf;
        input [63:0] test_name;
        begin
            @(posedge clk); // Wait for result
            @(posedge clk); // Pipeline delay
            
            if (result !== exp_res || zero !== exp_zero) begin
                $display("FAIL: %s - a=%h b=%h op=%h", test_name, a, b, op);
                $display("      Expected: result=%h zero=%b", exp_res, exp_zero);
                $display("      Got:      result=%h zero=%b", result, zero);
                tests_failed = tests_failed + 1;
            end else begin
                tests_passed = tests_passed + 1;
            end
        end
    endtask
    
    // Main test sequence
    initial begin
        // Initialize
        rst = 1;
        a = 0;
        b = 0;
        op = 0;
        valid_in = 0;
        
        // Reset
        repeat(3) @(posedge clk);
        rst = 0;
        valid_in = 1;
        @(posedge clk);
        
        // ===== ADD Tests =====
        $display("Testing ADD...");
        op = OP_ADD;
        
        // Basic add
        a = 8'h05; b = 8'h03;
        check_result(8'h08, 0, 0, "ADD basic");
        
        // Add with zero
        a = 8'h00; b = 8'h42;
        check_result(8'h42, 0, 0, "ADD zero");
        
        // Add resulting in zero
        a = 8'h00; b = 8'h00;
        check_result(8'h00, 1, 0, "ADD to zero");
        
        // Add overflow
        a = 8'hFF; b = 8'h01;
        check_result(8'h00, 1, 0, "ADD overflow");
        
        // ===== SUB Tests =====
        $display("Testing SUB...");
        op = OP_SUB;
        
        a = 8'h10; b = 8'h05;
        check_result(8'h0B, 0, 0, "SUB basic");
        
        a = 8'h05; b = 8'h05;
        check_result(8'h00, 1, 0, "SUB equal");
        
        // ===== AND Tests =====
        $display("Testing AND...");
        op = OP_AND;
        
        a = 8'hF0; b = 8'h0F;
        check_result(8'h00, 1, 0, "AND disjoint");
        
        a = 8'hFF; b = 8'hAA;
        check_result(8'hAA, 0, 0, "AND mask");
        
        // ===== OR Tests =====
        $display("Testing OR...");
        op = OP_OR;
        
        a = 8'hF0; b = 8'h0F;
        check_result(8'hFF, 0, 0, "OR combine");
        
        // ===== XOR Tests =====
        $display("Testing XOR...");
        op = OP_XOR;
        
        a = 8'hAA; b = 8'h55;
        check_result(8'hFF, 0, 0, "XOR pattern");
        
        a = 8'hFF; b = 8'hFF;
        check_result(8'h00, 1, 0, "XOR same");
        
        // ===== Shift Tests =====
        $display("Testing Shifts...");
        
        op = OP_SHL;
        a = 8'h01; b = 8'h03;
        check_result(8'h08, 0, 0, "SHL by 3");
        
        op = OP_SHR;
        a = 8'h80; b = 8'h02;
        check_result(8'h20, 0, 0, "SHR by 2");
        
        // ===== Compare Tests =====
        $display("Testing Compares...");
        
        op = OP_LT;
        a = 8'h05; b = 8'h10;
        check_result(8'h01, 0, 0, "LT true");
        
        a = 8'h10; b = 8'h05;
        check_result(8'h00, 1, 0, "LT false");
        
        op = OP_EQ;
        a = 8'h42; b = 8'h42;
        check_result(8'h01, 0, 0, "EQ true");
        
        a = 8'h42; b = 8'h43;
        check_result(8'h00, 1, 0, "EQ false");
        
        // ===== PASS Test =====
        $display("Testing PASS...");
        op = OP_PASS;
        a = 8'hDE; b = 8'hAD;
        check_result(8'hDE, 0, 0, "PASS");
        
        // Report
        #100;
        $display("");
        $display("========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Passed: %0d", tests_passed);
        $display("Failed: %0d", tests_failed);
        
        if (tests_failed == 0) begin
            $display("ALL TESTS PASSED");
            $finish(0);
        end else begin
            $display("SOME TESTS FAILED");
            $finish(1);
        end
    end
    
    // Timeout
    initial begin
        #10000;
        $display("TIMEOUT");
        $finish(1);
    end

endmodule

library ieee;
use ieee.std_logic_1164.all;

entity tb is
end tb;

architecture tb_arch of tb is

component TrafficLightsController is
 port(clk,rst,tr1,tr4 : in std_logic; r,g,y : out std_logic_vector(4 downto 0));
end component;

signal clk : std_logic := '0';
signal rst : std_logic := '0';
signal tr1 : std_logic := '0';
signal tr4 : std_logic := '0';

signal r : std_logic_vector(4 downto 0);
signal g : std_logic_vector(4 downto 0);
signal y : std_logic_vector(4 downto 0);

constant clk_period : time := 10 ns;

begin

inst1 : TrafficLightsController
port map(clk => clk, rst => rst, tr1 => tr1, tr4 => tr4, r => r, g => g, y =>y);

clk_process :process
begin
  clk <= '0';
  wait for clk_period/2;
  clk <= '1';
  wait for clk_period/2;
end process;
	
stim_proc: process
begin    
  rst <= '0';
  tr1 <= '0';
  tr4 <= '0';
  wait for clk_period*10;
  rst <= '1';
  tr1 <= '1';
  tr4 <= '0';
  wait;
end process;

end tb_arch;
library ieee;
use ieee.std_logic_1164.all;
use IEEE.STD_LOGIC_UNSIGNED.all;

entity TrafficLightsController is
 port(clk,rst,tr1,tr4 : in std_logic; r,g,y : out std_logic_vector(4 downto 0));
end TrafficLightsController;

architecture TrafficLightsController_arch of TrafficLightsController is

signal counter_1s: std_logic_vector(27 downto 0):= x"0000000";
signal delay_count:std_logic_vector(3 downto 0):= x"0";
signal clk_1s_enable: std_logic;
signal delay_gl0,delay_yl0,delay_gl1,delay_yl1,delay_gl2,delay_yl2,delay_gl3,delay_yl3,delay_gl4,delay_yl4: std_logic := '0';
signal enable_gl0,enable_yl0,enable_gl1,enable_yl1,enable_gl2,enable_yl2,enable_gl3,enable_yl3,enable_gl4,enable_yl4: std_logic :='0';
type FSM_States is (l2g,l2y,l3g,l3y,l4g,l4y,l0g,l0y,l1g,l1y);
signal cs,ns : FSM_States;

begin

process(clk,rst)
begin
    if(rst='0') then
	    cs <= l2g;
	 elsif(rising_edge(clk)) then
       cs <= ns;
	 end if;
end process;


process(cs,tr1,tr4,delay_gl0,delay_yl0,delay_gl1,delay_yl1,delay_gl2,delay_yl2,delay_gl3,delay_yl3,delay_gl4,delay_yl4)
begin
case cs is
when l2g =>
  enable_gl0 <= '0';
  enable_yl0 <= '0';
  enable_gl1 <= '0';
  enable_yl1 <= '0';
  enable_gl2 <= '1';
  enable_yl2 <= '0';
  enable_gl3 <= '0';
  enable_yl3 <= '0';
  enable_gl4 <= '0';
  enable_yl4 <= '0';
  r <= "11011";
  g <= "00100";
  y <= "00000";
 if(delay_gl2='1') then
   ns <= l2y;
 else 
   ns <= l2g; 
 end if;	

when l2y =>
  enable_gl0 <= '0';
  enable_yl0 <= '0';
  enable_gl1 <= '0';
  enable_yl1 <= '0';
  enable_gl2 <= '0';
  enable_yl2 <= '1';
  enable_gl3 <= '0';
  enable_yl3 <= '0';
  enable_gl4 <= '0';
  enable_yl4 <= '0';
  r <= "11011";
  g <= "00000";
  y <= "00100";
 if(delay_yl2='1') then
   ns <= l3g;
 else 
   ns <= l2y; 
 end if;	
	
when l3g =>
  enable_gl0 <='0';
  enable_yl0 <='0';
  enable_gl1 <='0';
  enable_yl1 <='0';
  enable_gl2 <='0';
  enable_yl2 <='0';
  enable_gl3 <='1';
  enable_yl3 <='0';
  enable_gl4 <='0';
  enable_yl4 <='0';
  r <= "10111";
  g <= "01000";
  y <= "00000";
 if(delay_gl3='1') then
   ns <= l3y;
 else 
   ns <= l3g; 
 end if;	

when l3y =>
  enable_gl0 <='0';
  enable_yl0 <='0';
  enable_gl1 <='0';
  enable_yl1 <='0';
  enable_gl2 <='0';
  enable_yl2 <='0';
  enable_gl3 <='0';
  enable_yl3 <='1';
  enable_gl4 <='0';
  enable_yl4 <='0';
  r <= "10111";
  g <= "00000";
  y <= "01000";
 if(delay_yl3='1') then
   ns <= l4g;
 else 
   ns <= l3y; 
 end if;	

when l4g =>
  enable_gl0 <='0';
  enable_yl0 <='0';
  enable_gl1 <='0';
  enable_yl1 <='0';
  enable_gl2 <='0';
  enable_yl2 <='0';
  enable_gl3 <='0';
  enable_yl3 <='0';
  enable_gl4 <='1';
  enable_yl4 <='0'; 
  r <= "01111";
  g <= "10000";
  y <= "00000";
 if(tr4 ='0') then
   ns <= l0g;
 elsif(delay_gl4='1') then
  ns <=  l4y;
 else 
   ns <= l4g; 
 end if;	

when l4y =>
  enable_gl0 <='0';
  enable_yl0 <='0';
  enable_gl1 <='0';
  enable_yl1 <='0';
  enable_gl2 <='0';
  enable_yl2 <='0';
  enable_gl3 <='0';
  enable_yl3 <='0';
  enable_gl4 <='0';
  enable_yl4 <='1'; 
  r <= "01111";
  g <= "00000";
  y <= "10000";
 if(delay_yl4='1') then
   ns <= l0g;
 else 
   ns <= l4y;
 end if;	

when l0g =>
  enable_gl0 <='1';
  enable_yl0 <='0';
  enable_gl1 <='0';
  enable_yl1 <='0';
  enable_gl2 <='0';
  enable_yl2 <='0';
  enable_gl3 <='0';
  enable_yl3 <='0';
  enable_gl4 <='0';
  enable_yl4 <='0'; 
  r <= "11110";
  g <= "00001";
  y <= "00000";
 if(delay_gl0='1') then
   ns <= l0y;
 else 
   ns <= l0g;  
 end if;

when l0y =>
  enable_gl0 <='0';
  enable_yl0 <='1';
  enable_gl1 <='0';
  enable_yl1 <='0';
  enable_gl2 <='0';
  enable_yl2 <='0';
  enable_gl3 <='0';
  enable_yl3 <='0';
  enable_gl4 <='0';
  enable_yl4 <='0'; 
  r <= "11110";
  g <= "00000";
  y <= "00001";
 if(delay_yl0='1') then
   ns <= l1g;
 else 
   ns <= l0y; 
 end if;

when l1g =>
  enable_gl0 <='0';
  enable_yl0 <='0';
  enable_gl1 <='1';
  enable_yl1 <='0';
  enable_gl2 <='0';
  enable_yl2 <='0';
  enable_gl3 <='0';
  enable_yl3 <='0';
  enable_gl4 <='0';
  enable_yl4 <='0'; 
  r <= "11101";
  g <= "00010";
  y <= "00000";
 if(tr1 ='0') then
   ns <= l2g;
 elsif(delay_gl1='1') then
   ns <= l1y; 
 else 
   ns <= l1g; 
 end if;	
	
when l1y =>
  enable_gl0 <='0';
  enable_yl0 <='0';
  enable_gl1 <='0';
  enable_yl1 <='1';
  enable_gl2 <='0';
  enable_yl2 <='0';
  enable_gl3 <='0';
  enable_yl3 <='0';
  enable_gl4 <='0';
  enable_yl4 <='0'; 
  r <= "11101";
  g <= "00000";
  y <= "00010";
 if(delay_yl1='1') then
   ns <= l2g;
 else 
   ns <= l1y; 
 end if;	
	
when others => ns <= l2g;
end case;
end process;


process(clk)
begin
if(rising_edge(clk)) then 
if(clk_1s_enable='1') then
 if(enable_gl0='1' or enable_yl0='1' or enable_gl1='1' or enable_yl1='1'or enable_gl2='1'or enable_yl2='1'or enable_gl3='1' or enable_yl3='1'or enable_gl4='1'or enable_yl4='1') then
   delay_count <= delay_count + x"1";
  if((delay_count = x"6") and enable_gl0='1') then
    delay_gl0 <= '1';
	 delay_yl0 <= '0';
	 delay_gl1 <= '0';
	 delay_yl1 <= '0';
	 delay_gl2 <= '0';
	 delay_yl2 <= '0';
	 delay_gl3 <= '0';
	 delay_yl3 <= '0';
	 delay_gl4 <= '0';
	 delay_yl4 <= '0';
    delay_count <= x"0";
  
  elsif((delay_count = x"1") and enable_yl0='1') then
    delay_gl0 <= '0';
	 delay_yl0 <= '1';
	 delay_gl1 <= '0';
	 delay_yl1 <= '0';
	 delay_gl2 <= '0';
	 delay_yl2 <= '0';
	 delay_gl3 <= '0';
	 delay_yl3 <= '0';
	 delay_gl4 <= '0';
	 delay_yl4 <= '0';
    delay_count <= x"0";
	 
  elsif((delay_count = x"3") and enable_gl1='1') then
    delay_gl0 <= '0';
	 delay_yl0 <= '0';
	 delay_gl1 <= '1';
	 delay_yl1 <= '0';
	 delay_gl2 <= '0';
	 delay_yl2 <= '0';
	 delay_gl3 <= '0';
	 delay_yl3 <= '0';
	 delay_gl4 <= '0';
	 delay_yl4 <= '0';
    delay_count <= x"0";

  elsif((delay_count = x"1") and enable_yl1='1') then
    delay_gl0 <= '0';
	 delay_yl0 <= '0';
	 delay_gl1 <= '0';
	 delay_yl1 <= '1';
	 delay_gl2 <= '0';
	 delay_yl2 <= '0';
	 delay_gl3 <= '0';
	 delay_yl3 <= '0';
	 delay_gl4 <= '0';
	 delay_yl4 <= '0';
    delay_count <= x"0";
	 
	 elsif((delay_count = x"6") and enable_gl2='1') then
    delay_gl0 <= '0';
	 delay_yl0 <= '0';
	 delay_gl1 <= '0';
	 delay_yl1 <= '0';
	 delay_gl2 <= '1';
	 delay_yl2 <= '0';
	 delay_gl3 <= '0';
	 delay_yl3 <= '0';
	 delay_gl4 <= '0';
	 delay_yl4 <= '0';
    delay_count <= x"0";
	 
	 elsif((delay_count = x"1") and enable_yl2='1') then
    delay_gl0 <= '0';
	 delay_yl0 <= '0';
	 delay_gl1 <= '0';
	 delay_yl1 <= '0';
	 delay_gl2 <= '0';
	 delay_yl2 <= '1';
	 delay_gl3 <= '0';
	 delay_yl3 <= '0';
	 delay_gl4 <= '0';
	 delay_yl4 <= '0';
    delay_count <= x"0";
	 
	 elsif((delay_count = x"6") and enable_gl3='1') then
    delay_gl0 <= '0';
	 delay_yl0 <= '0';
	 delay_gl1 <= '0';
	 delay_yl1 <= '0';
	 delay_gl2 <= '0';
	 delay_yl2 <= '0';
	 delay_gl3 <= '1';
	 delay_yl3 <= '0';
	 delay_gl4 <= '0';
	 delay_yl4 <= '0';
    delay_count <= x"0";
	 
	 elsif((delay_count = x"1") and enable_yl3='1') then
    delay_gl0 <= '0';
	 delay_yl0 <= '0';
	 delay_gl1 <= '0';
	 delay_yl1 <= '0';
	 delay_gl2 <= '0';
	 delay_yl2 <= '0';
	 delay_gl3 <= '0';
	 delay_yl3 <= '1';
	 delay_gl4 <= '0';
	 delay_yl4 <= '0';
    delay_count <= x"0";
	 
	 elsif((delay_count = x"3") and enable_gl4='1') then
    delay_gl0 <= '0';
	 delay_yl0 <= '0';
	 delay_gl1 <= '0';
	 delay_yl1 <= '0';
	 delay_gl2 <= '0';
	 delay_yl2 <= '0';
	 delay_gl3 <= '0';
	 delay_yl3 <= '0';
	 delay_gl4 <= '1';
	 delay_yl4 <= '0';
    delay_count <= x"0";
	 
	 elsif ((delay_count = x"1") and enable_yl4='1') then
    delay_gl0 <= '0';
	 delay_yl0 <= '0';
	 delay_gl1 <= '0';
	 delay_yl1 <= '0';
	 delay_gl2 <= '0';
	 delay_yl2 <= '0';
	 delay_gl3 <= '0';
	 delay_yl3 <= '0';
	 delay_gl4 <= '0';
	 delay_yl4 <= '1';
    delay_count <= x"0";
	 
	 else
	 delay_gl0 <= '0';
	 delay_yl0 <= '0';
	 delay_gl1 <= '0';
	 delay_yl1 <= '0';
	 delay_gl2 <= '0';
	 delay_yl2 <= '0';
	 delay_gl3 <= '0';
	 delay_yl3 <= '0';
	 delay_gl4 <= '0';
	 delay_yl4 <= '0';
	end if;
  end if;
 end if;
 end if;
end process;	
	 
process(clk)
begin
if(rising_edge(clk)) then 
 counter_1s <= counter_1s + x"0000001";
 if(counter_1s >= x"0000003") then
  counter_1s <= x"0000000";
 end if;
end if;
end process;
clk_1s_enable <= '1' when counter_1s = x"0003" else '0';


end TrafficLightsController_arch;
#!/bin/bash

# Nginx配置修复脚本
echo "🔧 修复 Nginx 配置..."

# 1. 检查当前状态
echo "📋 检查当前状态..."
echo "Nginx状态:"
sudo systemctl status nginx --no-pager -l

echo "当前启用的站点:"
ls -la /etc/nginx/sites-enabled/

# 2. 禁用默认站点
echo "🚫 禁用默认站点..."
sudo rm -f /etc/nginx/sites-enabled/default

# 3. 确保博客目录存在且有正确权限
echo "📁 检查博客目录..."
sudo mkdir -p /var/www/blog/myblog/public
sudo chown -R www-data:www-data /var/www/blog/
sudo chmod -R 755 /var/www/blog/

# 4. 重新创建博客Nginx配置
echo "📝 创建博客Nginx配置..."
sudo tee /etc/nginx/sites-available/blog > /dev/null << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    
    server_name _;
    
    root /var/www/blog/myblog/public;
    index index.html index.htm;
    
    # 日志文件
    access_log /var/log/nginx/blog_access.log;
    error_log /var/log/nginx/blog_error.log;
    
    location / {
        try_files $uri $uri/ =404;
    }
    
    # 静态文件缓存
    location ~* \.(css|js|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1M;
        add_header Cache-Control "public, immutable";
    }
    
    # 安全设置
    location ~ /\. {
        deny all;
    }
    
    # 错误页面
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
}
EOF

# 5. 启用博客站点
echo "✅ 启用博客站点..."
sudo ln -sf /etc/nginx/sites-available/blog /etc/nginx/sites-enabled/

# 6. 检查Nginx配置
echo "🔍 检查Nginx配置..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx配置检查通过"
    
    # 7. 重新加载Nginx
    echo "🔄 重新加载Nginx..."
    sudo systemctl reload nginx
    
    echo "✅ Nginx配置修复完成！"
else
    echo "❌ Nginx配置有错误，请检查"
    exit 1
fi

# 8. 检查博客文件是否存在
echo "📂 检查博客文件..."
if [ -f "/var/www/blog/myblog/public/index.html" ]; then
    echo "✅ 博客文件存在"
else
    echo "⚠️  博客文件不存在，需要生成"
fi

# 9. 显示状态
echo ""
echo "📊 当前状态:"
echo "Nginx状态: $(sudo systemctl is-active nginx)"
echo "启用的站点:"
ls -la /etc/nginx/sites-enabled/
echo ""
echo "🌐 现在可以通过浏览器访问你的服务器IP查看博客"
